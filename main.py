from sklearn.model_selection import train_test_split
import os, sys, time
import numpy as np

sys.path.append('C:\\Users\\psu61\\Documents\\Research\\top_audio_id\\audioId')

from gtda.diagrams import PersistenceEntropy, Scaler, BettiCurve
from gtda.homology import VietorisRipsPersistence
from gtda.homology import CubicalPersistence

from gtda.metaestimators import CollectionTransformer
from gtda.pipeline import Pipeline
from gtda.time_series import TakensEmbedding, SingleTakensEmbedding, takens_embedding_optimal_parameters


from audioId.core import audio, spectrogram
from audioId.pipeline import window_to_ph_vector
from audioId.pipeline import fingerprint_audio, compare_audios, get_matrix_from_dict, get_windows_from_audio, get_windows_from_values
from audioId.pipeline.match import get_delta_t_matching, get_error_from_matching
from audioId.ph import filtrations, ph
from audioId.ph.vectorization import BettiCurveExact
from audioId.transformations.transformation import MyTransformer, NoiseTransformer

from gtda.images import RadialFiltration, HeightFiltration


from sklearn.pipeline import make_union
from gtda.diagrams import NumberOfPoints, Amplitude
from sklearn.decomposition import PCA
from sklearn.utils import gen_batches

from tqdm.notebook import  tqdm


#from pl.lightning 


class GWAnalyzer(object):
    """docstring for GWAnalyzer"""

    SAMPLE_RATE = 4096

    def __init__(self, data_mixture, data_background):
        super(GWAnalyzer, self).__init__()
        self.data_mixture = data_mixture
        self.data_background  = data_background
        #self.N_samples = N_samples
        #self.truncate_length = truncate_length
        #self.slice_length = slice_length
        self.data = None
        self.label = None
        self.distance_from_merger = None #in seconds
        self.topological_features = None
        self.losses = None
        self.spectograms = None
        self.save_spectograms = False



        # RUN takens_embedding_optimal_parameters

        #self.time_delay
        #self.dimension
        #self.stride


    def preprocess(self, N_samples, truncate_length, slice_length, start_index=0):
        samples_multiplier  = int(truncate_length/slice_length)
        self.data =  np.concatenate([self.data_mixture[start_index:start_index+N_samples,:truncate_length].reshape(N_samples*samples_multiplier, slice_length), self.data_background[start_index:start_index+N_samples,:truncate_length].reshape(N_samples*samples_multiplier, slice_length)])
        self.label = np.concatenate([np.ones(N_samples*samples_multiplier), np.zeros(N_samples*samples_multiplier)])
        # Add Logic About distance_from_merger

        arr_sig = self.data_mixture - self.data_background



        self.distance_from_merger =  np.concatenate([((np.tile(np.argmax(arr_sig, axis=1)[start_index:start_index+N_samples],[samples_multiplier, 1]).T - (np.tile(np.arange(0,truncate_length, slice_length),[N_samples,1]) + slice_length))/4096).reshape(N_samples*samples_multiplier),((np.tile(np.argmax(arr_sig, axis=1)[start_index:start_index+N_samples],[samples_multiplier, 1]).T - (np.tile(np.arange(0,truncate_length, slice_length),[N_samples,1]) + slice_length))/4096).reshape(N_samples*samples_multiplier)])     
        pass

    #def graphNN(self):
    #   pass

    #def pointcloudNN(self):
    #   pass

    def spectogram_features(self):

        if self.data is None:
            raise Exception("Preprocessor not run yet")

        start = time.time()

        specs = []
        for wave in self.data:
            spec_stft = spectrogram.STFT.from_values(wave, self.SAMPLE_RATE)
            arr = np.abs(spec_stft.spec)
            arr /= arr.max()
            specs.append(arr)

        specs = np.array(specs)
        radial_filtration = RadialFiltration()
        cubical_persistence = CubicalPersistence(homology_dimensions=[0, 1, 2], n_jobs=-1)
        scaling = Scaler()

        metrics = [         
            {"metric": metric}
            for metric in ["bottleneck", "wasserstein", "landscape", "persistence_image"]
        ]


        feature_union = make_union(
                    PersistenceEntropy(normalize=True, nan_fill_value=-10),
                    NumberOfPoints(n_jobs=6),
                    *[Amplitude(**metric, n_jobs=6) for metric in metrics]
                )


        pipe = Pipeline(
            [
                ("filtration", radial_filtration),
                ("persistence", cubical_persistence),
                ("scaling", scaling),
                ("features", feature_union),
            ]
        )
        self.spectograms = specs

        feature_list = []
        for batch_idx in tqdm(gen_batches(specs.shape[0], batch_size=3400)):
            chunk = specs[batch_idx]
            print(f"Processing chunk of shape: {chunk.shape}")
            start = time.time()

            #features = topological_transfomer.fit_transform(self.data)
            features = pipe.fit_transform(chunk)
            feature_list.append(features)

            end = time.time()
            print(f'Chunk Elapsed Time: {end - start}')

        return np.vstack(feature_list)

    #print(f"features.shape {features.shape}")

    






    def spectogram_features_hist(self, n_bins):
        #extract features from the spectogram

        if self.data is None:
            raise Exception("Preprocessor not run yet")

        start = time.time()


        features = np.empty((0, 2*n_bins))

        fingerprint_params = dict(
            #spectrogramFct=spectrogram.MelSpectrogram,
            spectrogramFct=spectrogram.STFT,
            filter_fct=filtrations.intensity,
            compute_ph=ph.compute_ph_super_level,
            vect={0: BettiCurveExact(True), 1: BettiCurveExact(True)}
        )


        for wave in self.data:
            windows = get_windows_from_values(wave, spectrogram.STFT)
            bc = window_to_ph_vector(windows[(0.,1.0)], compute_ph=fingerprint_params["compute_ph"],
                         filter_fct=fingerprint_params["filter_fct"],
                         vect=fingerprint_params["vect"])



            #data = bc[0][0]
            #weights = bc[0][1]

            hist1, _ = np.histogram(bc[0][0], bins=np.linspace(0,1,n_bins+1), weights=bc[0][1]);
            hist2, _ = np.histogram(bc[1][0], bins=np.linspace(0,1,n_bins+1), weights=bc[1][1]);
            features = np.vstack([features, np.concatenate([hist1, hist2])])
            #bin_centers = (bins[:-1] + bins[1:]) / 2

        end = time.time()
        print(f'Spectogram Elapsed Time: {end - start}')



        return features

    def point_cloud_features(self, embedding_time_delay, embedding_dimension, stride):
        # Start with Takens Embedding, 

        if self.data is None:
            raise Exception("Preprocessor not run yet")
        
        embedder = TakensEmbedding(time_delay=embedding_time_delay,
                           dimension=embedding_dimension,
                           stride=stride)

        batch_pca = CollectionTransformer(PCA(n_components=3), n_jobs=6)

        persistence = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=18)
        persistence_2 = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], n_jobs=18, collapse_edges=True)

        scaling = Scaler()


        betticurve = BettiCurve(n_jobs=-1)

        entropy = PersistenceEntropy(normalize=True, nan_fill_value=-10)


        metrics = [         
            {"metric": metric}
            for metric in ["bottleneck", "wasserstein", "landscape", "persistence_image"]
        ]


        # Concatenate to generate 3 + 3 + (4 x 3) = 18 topological features
        feature_union = make_union(
            PersistenceEntropy(normalize=True, nan_fill_value=-10),
            NumberOfPoints(n_jobs=6),
            *[Amplitude(**metric, n_jobs=6) for metric in metrics]
        )


        pipe = Pipeline(
            [
                ("embedder", embedder),
                ("pca", batch_pca),
                ("persistence2", persistence_2),
                ("scaling", scaling),
                ("features", feature_union),
            ]
        )

        steps = [("embedder", embedder),
                    ("pca", batch_pca),
                    ("persistence", persistence),
                    ("scaling", scaling),
                    ("betticurve", betticurve)
                    #("entropy", entropy)
                    ]

        topological_transfomer = Pipeline(steps)


        chunks = np.array_split(self.data, 50, axis=0) 

        feature_list = []
        for batch_idx in tqdm(gen_batches(self.data.shape[0], batch_size=3400)):
            chunk = self.data[batch_idx]
            print(f"Processing chunk of shape: {chunk.shape}")
            start = time.time()

            #features = topological_transfomer.fit_transform(self.data)
            features = pipe.fit_transform(chunk)
            feature_list.append(features)

            end = time.time()
            print(f'Chunk Elapsed Time: {end - start}')

            #print(f"features.shape {features.shape}")

        return np.vstack(feature_list)

    def graph_features(self):

        if self.data is None:
            raise Exception("Preprocessor not run yet")
        pass


    def neuralnet_augmentation(self):

        # POINT CLOUD TRANSFORMER 
        pass


    def visualize(self):
        # PLOT 
        pass


    def classify(self):
        # Binary classification on Background vs Mixture
        if self.topological_features is None:
            raise Exception("Topololgical features not obtained yet")

        print("Begin Binary Classification with MLP")
            
        pass




    def obtain_topological_features(self, use_pointcloud, use_graph, use_spectogram):


        if self.data is None:
            raise Exception("Preprocessor not run yet")
            
        if use_pointcloud == False and use_graph == False and use_spectogram == False:
            raise Exception("At least one has to be true")

        _features = np.empty((self.data.shape[0], 0))


        if use_spectogram == True:
            _features = np.hstack([_features, self.spectogram_features()])
            self.save_spectograms = True

            #_features.append()

        if use_pointcloud == True:
            _features = np.hstack((_features, self.point_cloud_features(1,10,1).reshape(self.data.shape[0], -1)))


        if use_graph == True:
            #_features.append()
            pass

        


        print(f"Shape of the final features is {_features.shape}")
        self.topological_features = _features



    def save_data(self, path, name):
        if self.topological_features.any() == None:
            raise Exception("Topological Features Empty, can't save")

        if self.save_spectograms:
            with open(f'{os.path.join(path, name)}_spectograms.npy', 'wb') as f:
                np.save(f, self.spectograms)


        import pickle 


        with open(f'{os.path.join(path, name)}_labels.npy', 'wb') as f:
            np.save(f, self.label)

        with open(f'{os.path.join(path, name)}_distances.npy', 'wb') as f:
            np.save(f, self.distance_from_merger)



    def load_data(self, path, name):
        self.topological_features = np.load(f'{os.path.join(path, name)}_topofeatures.npy')
        self.label = np.load(f'{os.path.join(path, name)}_labels.npy')
        self.distance_from_merger = np.load(f'{os.path.join(path, name)}_distances.npy')
        pass



