import librosa
import numpy as np
import extract_feat as ef

local_config = {  
            'batch_size': 1, 
            'eps': 1e-5,
            'sample_rate': 22050,
            'load_size': 22050*20,
            'name_scope': 'SoundNet',
            'phase': 'extract',
            }

def analyse_sound(path):
    x, sr = librosa.load(path)
    td = librosa.get_duration(x)
    print("Time duration of audio: ", td)
    x = np.reshape(x, [1,-1,1,1])
    print("Shape of input waveform: ", x.shape)

    # Setup visible device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    # Load pre-trained model
    G_name = './models/sound8.npy'
    param_G = np.load(G_name, encoding = 'latin1').item()
            
    
    # Init. Session
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement=True
    sess_config.gpu_options.allow_growth = True
    
    with tf.Session(config=sess_config) as session:
        # Build model
        model = Model(session, config=local_config, param_G=param_G)
        init = tf.global_variables_initializer()
        session.run(init)
        
        model.load()
 
        features = ef.extract_feat(model, x, local_config)
        print("Shape of feature: ", features.shape)