from deep_insight.base import *
from deep_insight.data.crf_word2id import CrfWord2id
from deep_insight.k.compile import Compile
from deep_insight.k.train import TrainModel


class BilstmCrf(Base):

    def __init__(self, vocab_size=2513, EMBED_DIM=200, BiRNN_UNITS=200, tags_size=7):
        super(BilstmCrf, self).__init__()
        self.p('vocab_size', vocab_size)
        self.p('EMBED_DIM', EMBED_DIM)
        self.p('BiRNN_UNITS', BiRNN_UNITS)
        self.p('tags_size', tags_size)

    def run(self):
        param = self.params

        from tfos.nets.nlp.bilstm_crf import BilstmCrf
        # param = json.loads('<#zzjzParam#>')
        vocab_size = param.get('vocab_size')
        EMBED_DIM = param.get("EMBED_DIM")
        BiRNN_UNITS = param.get('BiRNN_UNITS')
        tags_size = param.get('tags_size')

        vocab_size = int(vocab_size)
        EMBED_DIM = int(EMBED_DIM)
        BiRNN_UNITS = int(BiRNN_UNITS)
        tags_size = int(tags_size)
        output_df = BilstmCrf(sqlc=sqlc).add(vocab_size, EMBED_DIM, BiRNN_UNITS, tags_size)
        outputRDD('<#zzjzRddName#>_BiLSTM_CRF_mode', output_df)
        output_df.show()


class TestInput(TestCase):
    def setUp(self) -> None:
        self.is_local = True
        self.data_dir = os.path.join(self.path, 'data/data/text/test_data.data')
        self.model_dir = os.path.join(self.path, 'data/model/bilstm_crf')
        self.vocab_size = 2513
        self.EMBED_DIM = 200
        self.BiRNN_UNITS = 200
        self.tags_size = 7

    @unittest.skip("")
    def test_bilstm_crf(self):
        BilstmCrf().run()
        SummaryLayer().run()

    # @unittest.skip("")
    def test_alex_net_v2_train(self):
        CrfWord2id(data_dir=self.data_dir).b(DATA_BRANCH).run()
        BilstmCrf(vocab_size=self.vocab_size, EMBED_DIM=self.EMBED_DIM, BiRNN_UNITS=self.BiRNN_UNITS,
                  tags_size=self.tags_size).b(MODEL_BRANCH).run()
        Compile(optimizer='adam', loss='crf_loss', metrics=['crf_accuracy']).run()
        TrainModel(
            input_prev_layers=MODEL_BRANCH,
            input_rdd_name=DATA_BRANCH,
            cluster_size=2,
            num_ps=1,
            batch_size=32,
            epochs=5,
            model_dir=self.model_dir,
            go_on='false').run()


if __name__ == '__main__':
    unittest.main()
