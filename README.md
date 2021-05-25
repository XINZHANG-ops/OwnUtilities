
  

# Xin-util

Simple self-use great functions

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install utilities.

```bash
pip install -i https://test.pypi.org/simple/ xin-util
```

## Usage

```python
# regular utilities  
from xin_util.PrettyPrintDict import pretty_print_dict
from xin_util.CreateDIYdictFromDataFrame import CreateDIYdictFromDataFrame
from xin_util.ReadWritePicklefile import read_pickle, save_pickle
from xin_util.DownloadFromS3 import S3_download_folder, S3_upload_folder
from xin_util.BarPlots import plot_stacked_bar
from xin_util.ZipAndUnzip import zip, unzip
from xin_util.AccessSQL import getDBData, create

# TimeSeries  
from xin_util.TimeSeriesFeatureEngineer import TimeSeries, create_average_feature

# NLP related  
from xin_util.TextProcess import text_tokens
from xin_util.ResamplingData import up_resample
from xin_util.Scores import single_label_f_score, single_label_included_score, multiple_label_included_score
from xin_util.TFIDFpredict import tf_idf_classify
from xin_util.EmbeddingCNNpredict import Convolution_Network_classify
from xin_util.EmbeddingRNNpredict import Recurrent_Network_classify
from xin_util.FASTTEXTpredict import fasttext_classify
from xin_util.NBpredict import Naive_Bayes_classify
from xin_util.ONEHOTNNpredict import Onehot_Network_classify
from xin_util.LatentDirichletAllocationClass import MyLDA

# Model training time prediction
from model_trainingtime_prediction.layer_level_utils import get_train_data
from model_trainingtime_prediction.random_network_gen import gen_nn
from model_trainingtime_prediction.train_time_predict import prediction_model
```
## Documentation
  * [PrettyPrintDict](#PrettyPrintDict)
  * [CreateDIYdictFromDataFrame](#CreateDIYdictFromDataFrame)
  * [ReadWritePicklefile](#ReadWritePicklefile)
  * [DownloadFromS3](#DownloadFromS3)
  * [ZipAndUnzip](#ZipAndUnzip)
  * [TimeSeriesFeatureEngineer](#TimeSeriesFeatureEngineer)
  * [TextProcess](#TextProcess)
  * [ResamplingData](#ResamplingData)
  * [Scores](#Scores)
  * [TFIDFpredict](#TFIDFpredict)
  * [EmbeddingCNNpredict](#EmbeddingCNNpredict)
  * [EmbeddingRNNpredict](#EmbeddingRNNpredict)
  * [FASTTEXTpredict](#FASTTEXTpredict)
  * [NBpredict](#NBpredict)
  * [ONEHOTNNpredict](#ONEHOTNNpredict)
  * [LatentDirichletAllocationClass](#LatentDirichletAllocationClass)
  * [AccessSQL](#AccessSQL)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)