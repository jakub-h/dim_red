library("tuneR");
original<-readWave("data/partB/the_entertainer.wav");
originalvector<-as.matrix(original@left);
nelements<-dim(originalvector)[1]*dim(originalvector)[2];
blocksize<-100;
nblocks<-floor(nelements/blocksize);
featuredata<-matrix(0,nblocks,blocksize);
for (blockindex in 1:nblocks)
{
  featuredata[blockindex,]<-t(originalvector[((blockindex-1)*blocksize+1):((blockindex-1)*blocksize+blocksize)]);
}

# -------------------your code begins here----------------
write.table(featuredata, file="data/partB/entertainer_featuredata.csv", row.names=FALSE, col.names=FALSE, sep=",")
write.table(originalvector, file="data/partB/entertainer_origvector.csv", row.names = FALSE, col.names = FALSE, sep = ",")
  # Insert your code here to compute a PCA reconstruction of the feature data into 
# a new matrix called reconstructed_featuredata
reconstructed_featuredata <- as.matrix(read.csv("data/partB/entertainer_reconstructed.csv", header = FALSE, sep = ","))
# -------------------your code ends here----------------


reconstructedvector<-matrix(0,nblocks*blocksize,1);
for (blockindex in 1:nblocks)
{
  reconstructedvector[((blockindex-1)*blocksize+1):((blockindex-1)*blocksize+blocksize)]<-t(reconstructed_featuredata[blockindex,]);
}

reconstructedaudio<-original
reconstructedaudio@left<-as.vector(reconstructedvector)

writeWave(normalize(reconstructedaudio, unit = "16"), "data/partB/the_entertainer_reconst.wav")
write.table(reconstructedvector, "data/partB/entertainer_reconstvector.csv", row.names = FALSE, col.names = FALSE, sep = ",")
  
