""" File containing the implementation of generators of data from images
    slicing them into patches
"""

from .processing import ImgProc
from .enums import ImgData

import glob
import numpy as np
import utils
from pathlib import Path
from itertools import repeat, cycle
from multiprocessing import Process as MultiProcess, Queue, Event as MultiEvent
from multiprocessing.pool import ThreadPool
from threading import Thread, Lock, Event as ThreadEvent
from collections import deque
from time import sleep
from setproctitle import setproctitle
from psutil import Process as PsProcess
from os import getpid
import pandas as pd
import os

# Thread shared variable for variables loaded
loaded_list = deque()
data_lock = Lock()
thread_stop_event = ThreadEvent()

class Generator:
    """ Class representing an object that analyses an image database and
    generate data from it """

    def __init__(self, input_shape, generator_conf, queue_mem=0.1):
        if not Path(generator_conf['path']).exists():
            raise FileNotFoundError(generator_conf['path'] + ' doesn\'t exist!')

        self.gen_conf = generator_conf.copy()
        self._input_shape = input_shape
        self._queue_mem = queue_mem
        self._gen_stop_event = None
        self._queue = None
        self._proc = None
        self._img_pathnames = []
        self._iterations = 0

    @staticmethod
    def _load_thread_task(files_list):
        """ Auxiliary task to load data asynchronously """
        global data_lock, loaded_list, thread_stop_event
        my_process = PsProcess(getpid())
        args = cycle(np.array_split(files_list,
                                    np.arange(20, len(files_list), 20)))
        del files_list
        load_pool = ThreadPool()
        init_memory = my_process.memory_percent()
        for arg in args:
            while my_process.memory_percent() - init_memory > 15:
                sleep(1)

            data = load_pool.starmap(ImgProc.load_image,
                                     zip(arg, repeat(ImgData.FLOAT)))
            data_lock.acquire()
            loaded_list.append(data)
            data_lock.release()

            if thread_stop_event.is_set():
                break

        data_lock.acquire()
        loaded_list.append([])
        data_lock.release()
        load_pool.close()
        load_pool.join()
        return

    @staticmethod
    def _get_next_chunk():
        """ Function that gets the data shared between the threads """
        while True:
            data_lock.acquire()
            if len(loaded_list) == 0:
                data_lock.release()
                sleep(1)
                continue
            break
        imgs = loaded_list.popleft()
        data_lock.release()
        return imgs

    def _data_task(self):
        """ The task of the database generator responsible for collecting
            data.
        """
        global data_lock, loaded_list, thread_stop_event
        setproctitle('python3 - data task')
        # Evaluation mode
        thread = Thread(target=self._load_thread_task,
                        args=(self._img_pathnames,))
        thread.start()
        iteration, patches = 0, []
        while not self._gen_stop_event.is_set():
            imgs = self._get_next_chunk()
            patches += [np.vstack(list(map(ImgProc.extract_img_patch, imgs,
                                           repeat(self._input_shape[1]))))]
            num_patches = np.sum(list(map(lambda e: len(e), patches)))
            if num_patches < self._input_shape[0]:
                continue
            patches = np.vstack(patches)
            indexes = np.arange(self._input_shape[0], len(patches) + 1,
                                self._input_shape[0])
            batches = list(filter(len, np.array_split(patches, indexes)))
            patches = [batches.pop(-1)] if len(batches) > 1 else []
            iteration += len(batches)
            if iteration > self._iterations:
                thread_stop_event.set()
                self._gen_stop_event.set()
                diff = iteration - self._iterations
                batches = batches[:-diff]
            list(map(self._queue.put, batches))
        patches = patches[0]
        pad_width = self._input_shape[0] - len(patches)
        pad_shape = [(0, pad_width)] + (len(patches.shape) - 1) * [(0, 0)]
        patches = np.pad(patches, pad_shape, 'symmetric')
        self._queue.put(patches)
        thread.join()
        data_lock.acquire()
        loaded_list.clear()
        data_lock.release()

    def get_db_files_pathnames(self):
        """ Method that returns the pathname of the files in the generator
            folders. It can contain non valid images. So you must test before
            opening them.
        """
        return self._img_pathnames

    def get_patch_size(self):
        """ Get the size of the patches """
        return self._input_shape[2]

    def get_db_folder(self):
        """ Get the folder of the database used """
        return Path(self.gen_conf['path'])

    def get_num_iterations(self):
        """ Get the number of iterations for this generator. This is true
            only in generators of training databases
        """
        return self._iterations

    def _create_index_file(self, index_pathname):
        """ Function that creates the index file of a folder """
        print('Indexing file does not exist! Indexing database!\n')
        pool = ThreadPool()
        all_files = np.array(
            sorted(glob.glob(self.gen_conf['path'] + '/**/*',
                             recursive=True)))
        valid, img_refs = list(zip(*pool.starmap(
            ImgProc.is_pillow_valid_img, zip(all_files))))
        valid = list(valid)
        img_refs = np.array(img_refs, dtype=object)[valid]
        width, height = list(zip(*pool.starmap(lambda i: i.size,
                                               zip(img_refs))))
        img_pathnames = list(map(os.path.relpath, all_files[valid],
                                 repeat(self.gen_conf['path'])))
        df = pd.DataFrame(zip(width, height),
                          index=pd.Index(img_pathnames, name='paths'),
                          columns=pd.Index(['width', 'height']))
        df.to_csv(index_pathname)
        pool.close(), pool.join()

    def _do_img_indexing(self):
        """ Function that analyses and writes a file describing all images
            in a folder with its size. If the file already exists, it reads
            and index img files in the folders
        """
        index_pathname = Path(self.gen_conf['path']) / 'index.csv'
        if not index_pathname.exists():
            self._create_index_file(index_pathname)
        data = pd.read_csv(index_pathname)
        paths, width, height = list(zip(*data.to_numpy()))
        self._img_pathnames = list(map(
            lambda p: Path(self.gen_conf['path']) / p, paths))
        self._img_pathnames = np.array(self._img_pathnames, dtype=object)

        if 'img_range' in self.gen_conf:
            if self.gen_conf['img_range']:
                r = np.arange(*self.gen_conf['img_range'])
                self._img_pathnames = self._img_pathnames[r]
                width = np.array(width)[r]
                height = np.array(height)[r]

        if not 'iterations' in self.gen_conf or \
                not self.gen_conf['iterations']:
            patches = list(map(ImgProc.calc_n_patches, zip(width, height),
                               repeat(self._input_shape[1])))
            num_patches = np.sum(patches)
            iterations = np.ceil(num_patches / self._input_shape[0]).astype(int)
        else:
            iterations = self.gen_conf['iterations']
        self._iterations = iterations

        if 'shuffle' in self.gen_conf:
            if self.gen_conf['shuffle']:
                np.random.shuffle(self._img_pathnames)

    def start(self):
        """ Procedures to start and wait for the generator """
        self._do_img_indexing()
        self._queue = Queue(utils.estimate_queue_size([self._input_shape],
                                                      [np.float32],
                                                      self._queue_mem))
        self._gen_stop_event = MultiEvent()
        self._proc = MultiProcess(target=self._data_task)
        self._proc.start()

    def stop(self):
        """ Stop the generator """
        if self._gen_stop_event:
            self._gen_stop_event.set()
        self._proc = None
        if self._proc:
            self._proc.close()
            self._proc.join()
        self._gen_stop_event = None
        if self._queue:
            self._queue.close()
        self._queue = None

    def get_batch(self):
        """ Get a batch from the queue """
        batch = self._queue.get()
        return batch
