###########################
### assume raw data sits in 'brain/REST_LR_roimean_STD.RData' and 'brain/REST2_LR_roimean_STD.RData'
###########################

library(reticulate)
np <- import("numpy")

#### process the first dataset
load("brain/REST_LR_roimean_STD.RData", mydata <- new.env())
np$savez("brain/data_p1.npz",
    ids=mydata$ids,
    labels=mydata$labels,
    label_names=mydata$label_names,
    map_names=mydata$map_names,
    x=mydata$roimean,
    z_ids=as.vector(mydata$pmat_score$ids),
    z_scores=as.vector(mydata$pmat_score$PMAT24),
    submap_node=as.vector(mydata$submap$Node),
    submap_network=as.vector(mydata$submap$Network))
write.csv(mydata$submap, 'brain/data_p1_submap.csv',row.names=FALSE)

#### process the second dataset
load("brain/REST2_LR_roimean_STD.RData", mydata2 <- new.env())
np$savez("brain/data_p2.npz",
    ids=mydata2$ids,
    labels=mydata2$labels,
    label_names=mydata2$label_names,
    map_names=mydata2$map_names,
    x=mydata2$roimean,
    z_ids=as.vector(mydata2$pmat_score$ids),
    z_scores=as.vector(mydata2$pmat_score$PMAT24),
    submap_node=as.vector(mydata2$submap$Node),
    submap_network=as.vector(mydata2$submap$Network))
write.csv(mydata2$submap, 'brain/data_p2_submap.csv',row.names=FALSE)
