This repository implements in Matlab the Regression Selection method described in
"Zhang, Huikun, Lee, Ching-pei, Ericksen, Spencer S., Mason, Blake J.,
Wlodarchak, Nathan, Mitchell, Julie C., Wildman, Scott A., Wright, Stephen J.,
Nowak, Robert, Gitter, Anthony, Hoffman, F. Michael, and Newton, Michael A.
Minimally-supervised, chemogenomic strategies for effective compound
prioritization on kinases."

This repository implements two experiments described in the manuscript:
1. Training a single model for prediction on new kinases

2. Conducting leave-one-out cross-validation and outputing the final
leave-one-out prediction on each input kinase


==================================================================================

For training the model, run the files
pkis1_1um_final.m and pkis2_final.m.

After finishing the script, the variable "G(1:end-1)" corresponds to the
compounds that are selected as the informer set, where G(end) corresponds to
the bias term that is set to be one on all kinases. Their corresponding
compound ids can be found in the files

cids_pkis1.mat and cids_pkis2.mat.

To predict the results on a new kinase, please group the activities on the
informer set of the new kinase into a dense vector y and then run

x = predict(y,Bsub,C,G);

==================================================================================

For conducting leave-one-out cross-validation, run the file

pkis1_1um.m.

Note that leave-one-out cross-validation is not conducted on pkis2 in the
manuscript.
