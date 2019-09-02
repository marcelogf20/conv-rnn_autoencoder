""" Implementation of image autoencoder and its dynamics. """


import tensorflow as tf
from pathlib import Path
import numpy as np
from multiprocessing import Process, Manager, Pool
import ctypes
from psutil import cpu_count
import time
import sys
import pandas as pd
import copy
from os import stat, walk, getpid
from itertools import repeat, product
from psutil import Process as PsProcess
from time import sleep
from setproctitle import setproctitle
import inspect

import utils
from .generator import Generator
from .clr import *
from .enums import *
from .processing import ImgProc


CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'


# TODO: investigate why recurrent network is having None gradients
# TODO: use a different gain network for each level
# TODO: implement a different lr for the gain network
# TODO: include gain in bitstream calculation
class AutoEnc:
    """ Class representing the image autoencoders. It has all methods necessary
        to operate with them.
    """
    ckpt_file_pattern = "vloss{:.4g}.ckpt"
    exec_folder = 'rec{rec}_res{res}_in{in_shape}_{time}'
    get_numpy = lambda t: t.numpy()

    class State:
        """ Class with useful information about the current execution of the
            autoencoder. It provides the information of the instantiated model
            currently running.
        """

        def __init__(self, summ_period=50):
            self.step = None
            self.global_step = []
            self.just_one_level = False
            self.autoenc_opt = []
            self.mode = None
            self.out_type = None
            self.res_weight = -1
            self.is_eval = None
            self.autoenc = []
            self.gain_net = None
            self.grad_ctx = None
            self.loss = None
            self.ckpt = None
            self.ref_pos = -1
            self.rec_factor = -1
            self.summ_period = summ_period
            self.var_names = []
            self.grad_names = []
            self.in_names = []
            self.outputs_names = []
            self.losses_names = []
            self.latents_names = []
            self.gain_out_names = []
            self.gain_var_names = []
            self.gain_grad_names = []
            self.tsb_writer = None
            self.out_queue = None

    def __init__(self, autoencoder_conf, run_conf, queue_mem=0.1):
        self._queue_mem = queue_mem
        self.auto_cfg = autoencoder_conf.copy()
        self.run_cfg = run_conf.copy()
        self.st = None
        self.generators = self._instantiate_generators()
        self._create_output_path()
        self.ckpt_path_pattern = str(
            self.out_name / Folders.CHECKPOINTS.name.lower() /
            self.ckpt_file_pattern)

    @staticmethod
    def _clear_last_lines(n=1):
        """ Clear the last n lines in stdout """
        for _ in range(n):
            sys.stdout.write(CURSOR_UP_ONE)
            sys.stdout.write(ERASE_LINE)

    def _instantiate_generators(self):
        """ Method to instantiate generator objects """
        shape = self.auto_cfg['input_shape']
        conf = self.run_cfg['generators']
        gen = {}
        if conf['train']['enabled']:
            gen['train'] = Generator(shape, conf['train'])
            gen['valid'] = Generator(shape, conf['valid'])
        if conf['test']['enabled']:
            gen['test'] = Generator(shape, conf['test'])
        return gen

    def _create_output_path(self):
        """ Method that creates the variable that stores the output path
            for the running model. It also creates the output root.
        """
        time_s = str(int(time.time()))
        res = int(self.auto_cfg['residues'])
        rec = int(self.auto_cfg['recursive'])
        in_shape = '{}.{}.{}.{}'.format(*self.auto_cfg['input_shape'])
        out_name = Path(
            self.exec_folder.format(rec=rec, res=res, in_shape=in_shape,
                                    time=time_s))
        self.out_name = self.run_cfg['out_folder'] / out_name
        self.out_name.mkdir(parents=True, exist_ok=True)

    def _create_names_for_summ(self):
        """ This method generates all names for the tensorboard summary """
        for cont, model in enumerate(self.st.autoenc):
            self.st.var_names.append(
                list(map(lambda x: x.name, model.variables)))
            net_grads = 'grads_' + str(cont) + '/{}'
            self.st.grad_names.append(
                list(map(lambda x: net_grads.format(x.name), model.variables)))
            self.st.in_names.append('inputs/input_' + str(cont))
            self.st.losses_names.append('losses/loss_' + str(cont))
            self.st.latents_names.append('latents/latent_' + str(cont))
            self.st.gain_out_names.append('gain_factor/gain_' + str(cont))
        self.st.losses_names.append('losses/mean_loss')
        self.st.gain_var_names = list(map(lambda x: x.name,
                                          self.st.gain_net.variables))
        self.st.gain_grad_names = list(map(
            lambda x: 'grads_gain/' + x.name, self.st.gain_net.variables))

    def _create_gain_net(self):
        """ This method creates a gain factor neural network based on configs.
            It's inspired in Toderici's idea
        """
        st, conf = self.st, self.auto_cfg
        input_shape = conf['input_shape'][1:]
        batch_size = conf['input_shape'][0]
        optimizer = Optimizers.from_string(conf['lr_politics']['optimizer'])

        input_t = tf.keras.Input(shape=input_shape, batch_size=batch_size)
        if conf['gain_net']['enabled']:
            tensor = input_t
            for layer_cfg in conf['gain_net']['layers']:
                layer = KLayers.from_string(layer_cfg.pop('type'))
                tensor = layer(**layer_cfg)(tensor)
            output = tensor
        else:
            output = KLayers.GET_ONES.value()(input_t)

        model = tf.keras.Model(inputs=input_t, outputs=output)
        st.gain_net = model
        st.gain_opt = optimizer(learning_rate=conf['lr_politics']['lr'])

    def _create_model(self):
        """ This method creates all objects necessary for running a model.
            Returns a checkpoint related to this model.
        """
        st, conf = self.st, self.auto_cfg
        optimizer = Optimizers.from_string(conf['lr_politics']['optimizer'])
        residues_range = range(conf['residues'])
        num_enc_layers = len(conf['enc_layers'])
        st.just_one_level = conf['recursive'] or conf['same_autoenc']
        if st.just_one_level:
            residues_range = range(1)

        batch_size = conf['input_shape'][0]
        eval_input = tf.keras.layers.Input(shape=(1,), batch_size=batch_size)
        for res in residues_range:
            layers_spec = list(map(
                copy.deepcopy, conf['enc_layers'] + conf['dec_layers']))

            with tf.name_scope('net_' + str(res)):
                input_t = tf.keras.Input(shape=conf['input_shape'][1:],
                                         batch_size=batch_size)
                tensor, output_t = input_t, []
                for cont, layer_cfg in enumerate(layers_spec):
                    layer = KLayers.from_string(layer_cfg.pop('type'))
                    if np.any(np.equal(layer, [KLayers.BINARIZE.value,
                                               KLayers.QUANTIZE.value])):
                        tensor = [tensor, eval_input]

                    tensor = layer(**layer_cfg)(tensor)

                    if cont == num_enc_layers - 1:
                        output_t.append(tensor)
                output_t.append(tensor)

            model = tf.keras.Model(inputs=[input_t, eval_input],
                                   outputs=output_t)
            st.autoenc.append(model)
            if conf['lr_politics']['cyclic'] and st.mode == ExecMode.TRAIN:
                st.global_step.append(tf.Variable(0, trainable=False))
                gen = self.generators[st.mode.name.lower()]
                gen.start()
                learning_rate = cyclic_learning_rate(
                    global_step=st.global_step[res],
                    mode=conf['lr_politics']['cyclic'],
                    learning_rate=conf['lr_politics']['lr'],
                    max_lr=5*conf['lr_politics']['lr'],
                    step_size=gen.get_num_iterations() / 20)
                gen.stop()
            elif conf['lr_politics']['schedule'] and st.mode == ExecMode.TRAIN:
                st.global_step.append(tf.Variable(0, trainable=False))
                gen = self.generators[st.mode.name.lower()]
                gen.start()
                schedule = Schedules.from_string(conf['lr_politics']
                                                 ['schedule'])
                learning_rate = schedule(conf['lr_politics']['lr'],
                                         st.global_step[res],
                                         gen.get_num_iterations()/200,
                                         0.96, staircase=True)
                gen.stop()
            else:
                learning_rate = conf['lr_politics']['lr']
                st.global_step.append(None)
            st.autoenc_opt.append(optimizer(learning_rate=learning_rate))

        if st.just_one_level:
            st.autoenc = conf['residues'] * st.autoenc
            st.autoenc_opt = conf['residues'] * st.autoenc_opt

        st.res_weight = 1. / len(st.autoenc)
        st.loss = Losses.from_string(conf['loss'])

        with tf.name_scope('gain_net'):
            self._create_gain_net()
        self._create_names_for_summ()
        st.ckpt = self._create_ckpt()

    def _create_ckpt(self, levels_to_consider=0):
        """ Create a checkpoint for the model """
        st = self.st
        if st.just_one_level:
            # Since in this case, all levels are just reference copies
            levels_ref = 1
        elif levels_to_consider > 0:
            levels_ref = levels_to_consider
        else:
            # Variable not specified, ckpt refers to all levels
            levels_ref = self.auto_cfg['residues']

        opts = st.autoenc_opt[:levels_ref]
        models = st.autoenc[:levels_ref]

        dict = {'opt' + str(i): o for i, o in enumerate(opts)}
        dict.update({'gain_opt': st.gain_opt})
        dict.update({'net' + str(i): m for i, m in enumerate(models)})
        dict.update({'gain_net': st.gain_net})
        ckpt = tf.train.Checkpoint(**dict)

        return ckpt

    def _create_out_folder(self):
        """ Auxiliary function to _handle_output that creates the prediction
            folder
        """
        if self.st.out_type == OutputType.NONE:
            return None

        pred_folder = self.out_name
        if self.st.mode == ExecMode.TRAIN:
            # TODO: treat this adequately when implementing serial training
            if self.st.out_type == OutputType.RESIDUES:
                pred_folder = OutputType.RESIDUES.name.lower()
        if self.st.mode == ExecMode.TEST:
            pred_folder /= Folders.TEST.name.lower()
        else:
            pred_folder /= Folders.VALIDATION.name.lower()
        pred_folder.mkdir(parents=True, exist_ok=True)

        return pred_folder

    @staticmethod
    def _perform_jpeg_img_routines(folder, orig_path, pos, net_bpp, jpeg_bpp,
                                   jpeg_metrics):
        """ Function that perform all output routines related to jpeg """
        try:
            cont = 0
            while not net_bpp[pos]:
                time.sleep(1)
                cont += 1
                if cont > 100:
                    frame_info = inspect.stack()[0]
                    print('Timeout in', frame_info.function, '-',
                          frame_info.lineno, 'while processing', orig_path,
                          end='\n\n')
                    return

            quality, metrics, bpp = list(zip(*list(map(
                AutoEnc._match_jpeg_quality, repeat(orig_path),
                net_bpp[pos]))))
            pts_to_pad = len(net_bpp[pos]) - len(bpp)
            quality, metrics, bpp = list(map(
                lambda a: np.pad(a, [0, pts_to_pad], 'symmetric'),
                [quality, metrics, bpp]))
            jpeg_bpp[pos] = list(bpp)
            [exec('ref[pos] = list(m)', {'ref': ref, 'pos': pos, 'm': m})
             for ref, m in zip(jpeg_metrics, metrics.swapaxes(0, 1))]

            folder /= str(Path(orig_path.name).with_suffix('.jpeg'))
            if not folder.exists():
                Path(folder).mkdir()

            # PIL doesn't handle numpy types
            quality = list(map(int, quality))
            quality_dict = list(map(lambda q: dict(quality=q), quality))
            out_path = list(map(AutoEnc.get_out_pathname, repeat(orig_path),
                                repeat(folder), repeat('.jpeg'),
                                range(len(net_bpp[pos]))))
            list(map(ImgProc.save_img_from_ref, repeat(orig_path), out_path,
                     quality_dict))
        except Exception as e:
            print(str(e), end='\n\n')

    @staticmethod
    def _perform_jpeg2k_img_routines(folder, orig_path, pos, net_bpp,
                                     jpeg2k_bpp, jpeg2k_metrics):
        """ Function that perform all output routines related to jpeg2k """
        try:
            folder /= Path(orig_path.name).with_suffix('.j2c')
            if not folder.exists():
                folder.mkdir()

            cont = 0
            while not net_bpp[pos]:
                time.sleep(1)
                cont += 1
                if cont > 100:
                    frame_info = inspect.stack()[0]
                    print('Timeout in', frame_info.function, '-',
                          frame_info.lineno, 'while processing', orig_path,
                          end='\n\n')
                    return

            metrics, bpp = list(map(np.array, zip(*list(map(
                AutoEnc._save_jpeg2000_image, repeat(orig_path), repeat(folder),
                net_bpp[pos], range(len(net_bpp[pos])))))))
            jpeg2k_bpp[pos] = list(bpp)
            [exec('ref[pos] = list(m)', {'ref': ref, 'pos': pos, 'm': m})
             for ref, m in zip(jpeg2k_metrics, metrics.swapaxes(0, 1))]
        except Exception as e:
            print(str(e), end='\n\n')

    @staticmethod
    def _save_out_analysis(img_paths, folder, bpps_proxy, metrics_proxy):
        """ Auxiliary function of _handle_output. It saves a csv containing
            the analysis for all images wrt all metrics for each codec.
        """
        def save_csv(data, csv_path, index, levels):
            names = ['bpp'] + list(map(lambda x: x.name.lower(), Metrics))
            cols = [x[0] + str(x[1]) for x in product(names, range(levels))]
            df = pd.DataFrame(data, index=pd.Index(index, name='img'),
                              columns=pd.Index(cols))
            df = df.sort_index()
            mean_df = pd.DataFrame(df.mean(axis=0).values.reshape(1, -1),
                                   columns=df.columns, index=pd.Index(['mean']))
            full_df = pd.concat((df, mean_df))
            full_df.to_csv(str(csv_path), float_format='%.5f')

        csv_path = list(map(
            lambda s: folder / ('_metrics_' + s.name.lower() + '.csv'), Codecs))
        # dims: (codecs, images, levels)
        bpps = np.array(list(map(list, bpps_proxy)))
        levels = bpps.shape[-1]
        # dims: (codecs, metrics, images, levels)
        metrics = np.array(list(map(list, metrics_proxy.flat))).reshape(
            (*list(metrics_proxy.shape), len(bpps[0]), -1))
        # dims: (codecs, images, metrics, levels)
        metrics = metrics.swapaxes(1, 2)
        # merge metrics and levels to just one dimension
        metrics = metrics.reshape((*list(metrics.shape[:-2]), -1))
        data = np.concatenate((bpps, metrics), axis=2)
        list(map(save_csv, data, csv_path, repeat(img_paths),
                 repeat(levels)))

    @staticmethod
    def _codecs_out_routines(pools, path, img_num, bpps, metrics,
                             latents, patches, out_folder):
        """ Auxiliary function of _handle_output. It does all routines necessary
            to the outputs and analysis of the codecs
        """
        pixel_num = np.prod(ImgProc.get_size(path))
        pools[0].apply_async(ImgProc.calc_bpp_using_gzip,
                             (latents, pixel_num,
                              bpps[Codecs.NET], img_num))
        pools[1].apply_async(ImgProc.save_img,
                             (path, out_folder / (path.stem + '_orig.png')))
        pools[2].apply_async(AutoEnc._save_imgs_from_patches,
                             (path, out_folder, patches, bpps[Codecs.NET],
                              metrics[Codecs.NET], img_num))
        pools[3].apply_async(AutoEnc._perform_jpeg_img_routines,
                             (out_folder, path, img_num,
                              bpps[Codecs.NET], bpps[Codecs.JPEG],
                              metrics[Codecs.JPEG]))
        pools[4].apply_async(AutoEnc._perform_jpeg2k_img_routines,
                             (out_folder, path, img_num,
                              bpps[Codecs.NET], bpps[Codecs.JPEG2K],
                              metrics[Codecs.JPEG2K]))
        pools[5].starmap_async(AutoEnc._save_metric_plot, zip(
            repeat(path), repeat(out_folder), metrics.swapaxes(0, 1),
            repeat(bpps), repeat(img_num), Metrics))

    @staticmethod
    def _instantiate_shared_variables(var_len):
        """ Auxiliary function that instantiate the variables maintained by
            the manager. It's used in _handle_output function
        """
        # positions: bpp, orig img, net, jpeg, jpeg2k, plots
        num_procs = np.array([1., .1, .35, .1, .35, .1])
        num_procs = np.clip(np.round(num_procs * cpu_count(), 0).astype(int),
                            1, cpu_count())
        pools = [Pool(n) for n in num_procs]
        manager = Manager()
        bpps = np.empty((len(Codecs),), dtype=object)
        bpps[:] = [manager.list([None] * var_len) for _ in range(len(Codecs))]
        metrics = np.empty((len(Codecs) * len(Metrics),), dtype=object)
        metrics[:] = [manager.list([None] * var_len)
                      for _ in range(len(Metrics) * len(Codecs))]
        metrics = metrics.reshape((len(Codecs), len(Metrics)))

        return pools, bpps, metrics

    def _handle_output(self):
        """ Routine executed to handle the output of the model """
        setproctitle('python3 - output task')
        gen = self.generators[self.st.mode.name.lower()]
        out_folder = self._create_out_folder()
        img_pathnames = list(gen.get_db_files_pathnames())
        curr_patches = [np.empty((self.auto_cfg['residues'], 0,
                                  *self.auto_cfg['input_shape'][1:]))]
        curr_latents, curr_gains = [], []
        if not self.st.out_type == OutputType.NONE:
            pools, bpps, metrics = self._instantiate_shared_variables(
                len(img_pathnames))

        my_process = PsProcess(getpid())
        all_processes = [my_process, *my_process.children()]
        get_mem = lambda *p: np.sum(list(map(lambda p2: p2.memory_percent(),
                                             p)))
        init_memory = get_mem(*all_processes)
        stop = False
        for file_num, curr_file in enumerate(img_pathnames):
            while get_mem(*all_processes) - init_memory > 15:
                sleep(2)

            n_patches = ImgProc.calc_n_patches(curr_file, gen.get_patch_size())
            patches_count = len(curr_patches[0][0])
            while patches_count < n_patches:
                model_data, model_latent, model_gain = self.st.out_queue.get()
                if not model_data:
                    stop = True
                    break
                curr_patches.append(model_data)
                curr_latents.append(model_latent)
                curr_gains.append(model_gain)
                patches_count += len(curr_patches[-1][0])

            if stop and patches_count < n_patches:
                break

            curr_patches = np.concatenate(curr_patches, axis=1)
            patches, curr_patches = np.array_split(curr_patches, [n_patches], 1)
            curr_patches = [curr_patches]
            if self.st.out_type == OutputType.NONE:
                curr_latents.clear(), curr_gains.clear()
                continue

            if not self.auto_cfg['recursive']:
                patches = np.cumsum(patches, axis=0)
            patches = np.clip(patches, *ImgData.FLOAT.value)
            patches = ImgProc.conv_data_format(patches, ImgData.UBYTE)

            curr_latents = np.concatenate(curr_latents, axis=1)
            latents, curr_latents = np.array_split(curr_latents, [n_patches], 1)
            curr_latents = [curr_latents]
            latents = latents.reshape((latents.shape[0], -1))

            curr_gains = np.concatenate(curr_gains, axis=1)
            gains, curr_gains = np.array_split(curr_gains, [n_patches], 1)
            curr_gains = [curr_gains]

            AutoEnc._codecs_out_routines(pools, curr_file, file_num, bpps,
                                         metrics, latents, patches, out_folder)

        if not self.st.out_type == OutputType.NONE:
            list(map(lambda p: p.close(), pools))
            list(map(lambda p: p.join(), pools))
            AutoEnc._save_out_analysis(img_pathnames, out_folder, bpps, metrics)

    @staticmethod
    def _save_all_jpeg_images(img_path, pred_folder, quality, psnr, bpp,
                              level_array):
        """ Auxiliary function that receives an array of qualities and save
            the corresponding jpeg images
        """
        folder = pred_folder / str(Path(img_path.name).with_suffix('.jpeg'))
        if not folder.exists():
            Path(folder).mkdir()

        # PIL doesn't handle numpy types
        quality = list(map(int, quality))
        quality_dict = list(map(lambda q: dict(quality=q), quality))

        out_path = list(map(AutoEnc.get_out_pathname, repeat(img_path),
                            repeat(folder), repeat('.jpeg'), psnr, bpp,
                            level_array))
        list(map(lambda *arg: ImgProc.save_img_from_ref(*arg[0:-1], **arg[-1]),
                 repeat(img_path), out_path, quality_dict))

    @staticmethod
    def _save_imgs_from_patches(orig_path, save_folder, patches_list,
                                bpp_proxy, metrics_proxy, pos, color='RGB'):
        """ Function that gets the predicted patches, and reconstruct the image.
        """
        try:
            orig_ref = ImgProc.load_image(orig_path, ImgData.UBYTE, color)
            img_size = ImgProc.get_size(orig_path)

            new_img = map(lambda p, s: ImgProc.reconstruct_image(p, *s),
                          patches_list, repeat(img_size))
            new_img = list(map(ImgProc.conv_data_format, new_img,
                               repeat(ImgData.UBYTE)))
            for metric in Metrics:
                metrics_proxy[metric][pos] = list(map(ImgProc.calc_metric,
                                                      repeat(orig_ref), new_img,
                                                      repeat(metric)))
            cont = 0
            while not bpp_proxy[pos]:
                time.sleep(1)
                if cont > 100:
                    frame_info = inspect.stack()[0]
                    print('Timeout in', frame_info.function, '-',
                          frame_info.lineno, 'while processing', orig_path,
                          end='\n\n')
                    return
            new_path = map(AutoEnc.get_out_pathname, repeat(orig_path),
                           repeat(save_folder), repeat('.png'),
                           np.arange(len(bpp_proxy[pos])))
            list(map(ImgProc.save_img, new_img, new_path, repeat(color)))
        except Exception as e:
            print(str(e), end='\n\n')

    @staticmethod
    def _match_jpeg_quality(orig_img, orig_bpp):
        """ Auxiliary function of _perform_jpeg_img_routines. Get the jpeg
            corresponding quality considering the bpp passed as parameter.
        """
        csv = orig_img.parent / 'jpeg'
        csv /= Path(orig_img.name).with_suffix('.jpeg.csv')
        df = pd.read_csv(csv)
        bpp = df.loc[:, 'bpp'].values
        matching_index = np.argsort(np.abs(bpp - orig_bpp))[0]
        matching_pos = np.unique(matching_index)
        metrics = []
        list(map(lambda x: metrics.append(df.loc[:, x.name.lower()].
                                          values[matching_pos][0]),
                 Metrics))
        # PIL doesn't handle jpeg quality as numpy int
        quality = int(df.loc[:, 'quality'].values[matching_pos][0])
        bpp = bpp[matching_pos][0]
        return quality, metrics, bpp

    @staticmethod
    def _save_jpeg2000_image(img_path, folder, bpp, level=-1):
        """ Save a image reference in kakadu jpeg2000 format """
        bmp_path = folder / ('level' + str(level) + '.bmp')
        ImgProc.save_img_from_ref(img_path, bmp_path)
        pixels_num = np.prod(ImgProc.get_size(img_path))

        aux_out_path = AutoEnc.get_out_pathname(
            img_path, folder, '.j2c', level=level)
        ImgProc.save_jpeg2000_kakadu(bmp_path, aux_out_path, bpp)
        j2k_metrics = list(map(ImgProc.calc_metric, repeat(img_path),
                               repeat(aux_out_path), Metrics))
        j2k_bpp = 8 * stat(aux_out_path).st_size / pixels_num
        out_path = AutoEnc.get_out_pathname(
            img_path, folder, '.j2c', level)
        aux_out_path.rename(out_path)
        bmp_path.unlink()
        return j2k_metrics, j2k_bpp

    @staticmethod
    def _save_metric_plot(img_path, folder, metrics_proxy, bpp_proxy, pos,
                          metrics):
        """ Method that saves a scatter plot related to an image """
        # To make matplotlib multiprocessing compatible
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter

        try:
            for metric, bpp in zip(metrics_proxy, bpp_proxy):
                cont = 0
                while not metric[pos] or not bpp[pos]:
                    time.sleep(1)
                    cont += 1
                    if cont > 100:
                        frame_info = inspect.stack()[0]
                        print('Timeout in', frame_info.function, '-',
                              frame_info.lineno, 'while processing', img_path,
                              end='\n\n')
                        return

            metric = list(map(lambda e: e[pos], metrics_proxy))
            bpp = list(map(lambda e: e[pos], bpp_proxy))

                fig, ax = plt.subplots()
                plt.xlabel('bpp')
                plt.ylabel(metrics.name.lower())
                plt.grid(True)
                for curr_bpp, curr_metric in zip(bpp, metric):
                    plt.plot(curr_bpp, curr_metric,
                             marker='o', markersize=6, linewidth=2)
                legend = list(map(lambda s: s.name.lower(), Codecs))
                plt.legend(legend, loc='upper left')
                img_size = ImgProc.get_size(img_path)
                plt.title(str(img_path.name) + ', {} x {}'.format(*img_size))

                all_bpp = np.sort(np.unique(np.concatenate(bpp)))
                all_metric = np.sort(np.unique(np.concatenate(metric)))
                min_bpp, max_bpp = min(all_bpp), max(all_bpp)
                min_metric, max_metric = min(all_metric), max(all_metric)
                plt.xticks(np.arange(min_bpp, max_bpp, 0.05))
                plt.xticks(rotation=90)
                plt.yticks(np.arange(min_metric, max_metric))

                if min_bpp != max_bpp:
                    plt.xlim(min_bpp, max_bpp)
                if min_metric != max_metric:
                    plt.ylim(min_metric, max_metric)
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

                fig.set_size_inches(*(fig.get_size_inches() * 1.5))
                plt.tight_layout()
                plot_name = folder / (img_path.stem + '_plot_' +
                                      metrics.name.lower() +
                                      Path(img_path.name).suffix)
                plt.savefig(str(plot_name), dpi=180)
                plt.close()
        except Exception as e:
            print(str(e), end='\n\n')

    @staticmethod
    def get_out_pathname(img_path, save_folder, ext='.png', level=-1):
        """ Construct an output pathname based on original image path """
        img_path = Path(img_path)
        save_folder = Path(save_folder)
        level_str = ''
        if level >= 0:
            level_str = '_level{:02.0f}'.format(level)

        new_name = img_path.stem + level_str + ext
        save_name = save_folder / new_name
        return save_name

    def log_distrib(self, var, context):
        """ Method that logs distributions in tensorboard """
        with tf.name_scope(context):
            tf.contrib.summary.histogram('hist', var, step=self.st.step)

    def log_loss(self, var, context):
        """ Specific method for logging the value of the loss to tensorboard """
        with tf.name_scope(context):
            tf.contrib.summary.scalar('value', var, step=self.st.step)

    # TODO: test for recursive network
    def _net_loop(self, data):
        """ Model loop when the model is not recursive and there's no
            weight sharing.
        """
        st = self.st
        loss_list, output_list, latent_list, autoenc_grads = [], [], [], []
        gain_grads, gain_list = [], []
        input_list = [tf.constant(data, dtype=tf.float32)]
        rec_t = tf.zeros_like(input_list[-1])
        mean_loss = tf.constant(0.)
        for l, autoenc in enumerate(st.autoenc):
            with st.grad_ctx() as tape:
                gain_list.append(st.gain_net(input_list[-1]))
                input_list[-1] *= gain_list[-1]
                latent, output = autoenc([input_list[-1], st.is_eval])
                output /= gain_list[-1]
                input_list[-1] /= gain_list[-1]
                loss = st.loss(input_list[st.ref_pos], output)
                mean_loss += st.res_weight * loss

            loss_list.append(loss)
            output_list.append(output[:])
            latent_list.append(latent[:])
            rec_t = st.rec_factor * rec_t + output
            input_list.append(data - rec_t)

            if not st.just_one_level and st.mode == ExecMode.TRAIN:
                autoenc_grads.append(tape.gradient(loss, autoenc.variables))

        if st.mode == ExecMode.TRAIN:
            if st.just_one_level:
                autoenc_grads.append(tape.gradient(mean_loss,
                                                   st.autoenc[0].variables))
            if self.auto_cfg['gain_net']['enabled']:
                gain_grads = tape.gradient(mean_loss, st.gain_net.variables)
                st.gain_opt.apply_gradients(zip(gain_grads,
                                                st.gain_net.variables))
        list(map(lambda opt, grads, model, step:
                 opt.apply_gradients(zip(grads, model.variables), step),
                 st.autoenc_opt, autoenc_grads, st.autoenc, st.global_step))

        list(map(lambda names, mod: list(map(
            self.log_distrib, mod.variables, names)), st.var_names, st.autoenc))
        list(map(lambda names, grads: list(map(
            self.log_distrib, grads, names)), st.grad_names, autoenc_grads))
        list(map(self.log_distrib, latent_list, st.latents_names))
        list(map(self.log_distrib, input_list, st.in_names))
        list(map(self.log_distrib, gain_list, st.gain_out_names))
        list(map(self.log_distrib, st.gain_net.variables, st.gain_var_names))
        list(map(self.log_distrib, gain_grads, st.gain_grad_names))

        loss_list = [mean_loss] + loss_list
        list(map(self.log_loss, loss_list, self.st.losses_names))
        loss_list = list(map(AutoEnc.get_numpy, loss_list))
        output_list = list(map(AutoEnc.get_numpy, output_list))
        latent_list = list(map(AutoEnc.get_numpy, latent_list))
        gain_list = list(map(AutoEnc.get_numpy, gain_list))

        if self.st.out_type == OutputType.RESIDUES:
            output_list = list(map(AutoEnc.get_numpy, input_list))

        return loss_list[0], output_list, latent_list, gain_list

    def _do_jpeg_analysis(self):
        """ Method that makes jpeg analysis in the current database. In the
            case of jpeg it's useful because the codec works with nominal
            quality, not one directly converted into psnr / bpp
        """
        gen = self.generators[self.st.mode.name.lower()]
        file_paths = gen.get_db_files_pathnames()
        gen_folder = gen.get_db_folder()
        pool = Pool()
        walk_ret = list(walk(gen_folder))
        root = Path(walk_ret[0][0])
        s_folders = [root] + list(map(lambda f: root / f, walk_ret[0][1]))
        s_folders = list(filter(lambda s: s.name != 'jpeg', s_folders))
        s_folders = list(map(lambda f: f / 'jpeg', s_folders))
        list(map(lambda x: x.mkdir(exist_ok=True), s_folders))

        csv = np.array(list(map(
            lambda p: p.parent / 'jpeg' / (p.stem + '.jpeg.csv'), file_paths)))
        to_create_index = list(map(lambda p: not p.exists(), csv))
        csv = csv[to_create_index]
        file_paths = file_paths[to_create_index]
        pool.starmap_async(ImgProc.save_jpeg_analysis, zip(file_paths, csv))
        pool.close()
        print('Analyzing the generator folder with jpeg.', end='\n\n')
        pool.join()

    def _run_model(self):
        """ Generic function that executes the current model based on the
            parameters passed
        """
        setproctitle('python3 - network')
        st = self.st
        st.is_eval = tf.constant(not st.mode == ExecMode.TRAIN)
        st.ref_pos, st.rec_factor = -1, 1
        if self.auto_cfg['recursive']:
            st.ref_pos, st.rec_factor = 0, 0

        tsb_log = tf.contrib.summary.record_summaries_every_n_global_steps
        gen = self.generators[st.mode.name.lower()]
        gen.start()

        if st.mode == ExecMode.TRAIN:
            st.grad_ctx = lambda: tf.GradientTape(persistent=True)
        else:
            st.grad_ctx = lambda: memoryview(b'')
            self._do_jpeg_analysis()

        shape = list(map(
            lambda out: [len(self.st.autoenc)] + out.shape.as_list(),
            self.st.autoenc[0].outputs + self.st.gain_net.outputs))
        dtype = (len(self.st.autoenc) + 1) * [np.float32]
        st.out_queue = Manager().Queue(
            utils.estimate_queue_size(shape, dtype, self._queue_mem))
        patch_proc = Process(target=AutoEnc._handle_output, args=(self, ))
        patch_proc.start()

        log_dir = self.out_name / Folders.TENSORBOARD.name.lower() \
                  / self.st.mode.name.lower()
        self.st.tsb_writer = tf.contrib.summary.create_file_writer(
            str(log_dir))

        # Execution of the model
        iter_str = '{:d}/' + str(gen.get_num_iterations()) + ': {}'
        mean_loss = 0.
        for st.step in range(1, gen.get_num_iterations() + 1):
            data = gen.get_batch()
            with st.tsb_writer.as_default(), tsb_log(st.summ_period, st.step):
                loss, outputs, latents, gains = self._net_loop(data)
                if self.auto_cfg['recursive']:
                    self.st.autoenc[0].reset_states()
                st.out_queue.put([outputs, latents, gains])
                AutoEnc._clear_last_lines()
                print(iter_str.format(st.step, str(loss)))
                mean_loss += loss

        st.out_queue.put([[], [], []])
        mean_loss /= st.step
        patch_proc.join()
        st.tsb_writer.close()
        gen.stop()

        return mean_loss

    def test_model(self):
        """ Evaluate the eager model for validation or testing """
        if not self.st:
            self.st = self.State()
            self._create_model()
            l_ckpt = self._create_ckpt(self.run_cfg['load']['residues'])
            l_ckpt.restore(self.run_cfg['load']['ckpt'])
        self.st.epoch_str = ''
        self.st.mode = ExecMode.TEST
        self.st.out_type = OutputType.RECONSTRUCTION

        print('\nTESTING:')
        self._run_model()

    # TODO: incorporate possibility of validation steps between iterations
    def train_model(self):
        """ Train the model using the eager execution """
        self.st = self.State()
        self.st.mode = ExecMode.TRAIN
        self._create_model()
        l_ckpt = self._create_ckpt(self.run_cfg['load']['residues'])
        l_ckpt.restore(self.run_cfg['load']['ckpt'])
        print('\nTRAINING:')
        self.st.mode = ExecMode.TRAIN
        self.st.out_type = OutputType.NONE
        self._run_model()
        print('\nVALIDATING:')
        self.st.mode = ExecMode.VALID
        self.st.out_type = OutputType.RECONSTRUCTION
        mean_loss = self._run_model()
        self.st.ckpt.save(self.ckpt_path_pattern.format(mean_loss))
