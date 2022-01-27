# RESERVE
Previously known as RESCUE (Renewable Energy Salient Combined Uncertainty Estimator)

## *Legal Information:*
This file is part of the E3 RESERVE Model.

Copyright (C) 2021 Energy and Environmental Economics, Inc.
For contact information, go to www.ethree.com

The E3 RESERVE Model is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The E3 RESERVE Model is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with the E3 RESERVE Model (in the file LICENSE.TXT). If not,
see <http://www.gnu.org/licenses/>.

## *Description:*
A machine-learning based framework to quantify the short-term uncertainty in netload forecast developed by E3. 
The main structure of the model include a multi-layer artificial neural net with the pinball loss function as objective. Conditional on combinations of input, the model is able to output quantile forecast for the net-load forecast error.

This notebook contains the work stream of ingesting pre-processed data, set up cross validation folds, 
training and deployment, and calling functions for diagnostics. 
For detailed implementation of data preprocessing or quoted functions, please refer to other script files. 

RESCUE model supports multi-objective learning. For example, in addition to producing the quantile forecast of Net Load forecast error, it can be trained to simultaneously predict the Load, Solar and Wind forecast error. The objectives can be weighted per user's judgement of relative importance.


## *Highlights:*
1. Incorporating a wide gamut of information: weather, calendar, forecast, and lagged error aware. 
2. Intrinsically handles resource correlation as solar,wind, and load errors are co-trained within the model.
3. Produces multiple prediction intervals for expected error in netload, load, solar and wind forecasting, for cherry picking down-stream
4. Model agnostic. No requirement on knowledge of the inner workings of the netload forecast
5. Adheres to best practice in statistics: cross validation, normalization, early-stopping, etc.

## *User Guide:*
For a user guide, please refer to this [document](https://willdan.box.com/s/4ma9j120bfxj55p58jzltqnquto31ari)
