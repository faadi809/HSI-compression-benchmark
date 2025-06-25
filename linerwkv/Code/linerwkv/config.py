class Config(object):
    def __init__(self):
        # Dataset
        self.dataset_dir = "/home/data/Fahad/codes/codes_for_Review/linerwkv-main"
        self.dataset_difficulty = "paviau"
        self.device = "cuda"

        # PaviaU dimensions
        self.width = 610
        self.height = 340
        self.bands = 103
        self.dtype = "uint16"  # BSQ pixel type

        # Optional: patching and split control
        self.train_split = 0.8  # 80% for training
        self.random_split = True  # shuffle patches before splitting

        # Architecture
        self.dim_enc = 32
        self.dropout = 0
        self.kernel_size = 3
        self.N_layers_encoder = 2
        self.n_layer_lines = 2
        self.n_layer_bands = 2
        self.ctx_len = 128
        self.dim_att = 32
        self.dim_ffn = 32
        self.tiny_att_layer = -1
        self.tiny_att_dim = 0
        self.N_layers_decoder = 2
        self.residual = False
        self.num_workers = 50

        # Learning
        self.batch_size = 4
        self.pos_size = 64
        self.epoch_count = 500
        self.layerwise_lr = 1
        self.weight_decay = 0
        self.weight_decay_final = 0
        self.lr_init = 5e-4
        self.lr_final = 1e-6
        self.warmup_steps = 50
        self.betas = (0.9, 0.99)
        self.adam_eps = 1e-8
        self.epoch_save = 10
        self.channels_subset = None
	#for lossless
        self.lossless = True  # add this
        self.delta = 0        # quantization step (0 for lossless)

