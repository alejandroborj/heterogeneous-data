suppressMessages(library(KnowSeq))
library(caret)
#if (!require("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("KnowSeq")

library(KnowSeq)
library(glue)


create_dataset <- function(SPLIT, mode) {
  gdc_dataset <- read.csv(glue("D:/data/RNASeq/dataset/labels/TCIAdataset_{mode}{SPLIT}.csv", sep = ""))
  #gtex_dataset <- read.csv(glue("D:/data/RNASeq/dataset/labels/TCIAdataset_{mode}{SPLIT}.csv", sep = ""))
  
  case_ids <- gdc_dataset$Sample.ID#as.vector(append(gdc_dataset$Sample.ID, gtex_dataset$Sample.ID))
  labels <- gdc_dataset$Class##as.vector(append(gdc_dataset$Class, gtex_dataset$Class))
  
  #print(labels)
  
  countsMatrix <- read.csv(paste(glue("D:/data/RNASeq/dataset/countsMatrix/countsMatrixTCIA_{mode}_split"), SPLIT,".csv", sep = ""),row.name=1)
  countsMatrix <- as.matrix(countsMatrix)
  
  # Downloading human annotation
  myAnnotation <- getGenesAnnotation(rownames(countsMatrix))
  
  # Calculating gene expression values matrix using the counts matrix
  
  expressionMatrix <- calculateGeneExpressionValues(countsMatrix, myAnnotation, genesNames = TRUE)
  
  # library(preprocessCore)
  # df_norm <- as.data.frame(normalize.quantiles(expressionMatrix))
  
  # write.csv(df_norm,"D:/data/RNASeq/dataset/pre_removal.csv", row.names=TRUE)
  
  # training
  
  svaMod <- batchEffectRemoval(expressionMatrix, labels, method = "sva")
  
  # df_norm <- as.data.frame(normalize.quantiles(expressionMatrix))
  
  #print(expressionMatrix)
  
  DEGsInformation <- DEGsExtraction(expressionMatrix, labels, lfc = 2,
                                         pvalue = 0.01, number = Inf, 
                                         cov = 2)#svaCorrection = FALSE,
  
  #print(DEGsInformation)
  
  DEGsInformation <- DEGsInformation$DEG_Results
  
  topTable <- DEGsInformation$DEGs_Table
  DEGsMatrix <- DEGsInformation$DEGs_Matrix
  
  #write.csv(DEGsMatrix,"D:/data/RNASeq/dataset/post_removal.csv", row.names=TRUE)
  
  DEGsMatrixML <- DEGsMatrix
  #DEGsMatrixML <- DEGsMatrixML[, 1:dim(DEGsMatrixML)[2]-1]
  
  DEGsMatrixML <- t(DEGsMatrixML)
  
  # Feature selection process with mRMR and RF
  
  
  mrmrRanking <-featureSelection(DEGsMatrixML,labels,colnames(DEGsMatrixML), mode = "mrmr")
  
  save_df <- as.data.frame(DEGsMatrixML, row.names = case_ids) 
  reorder_df <- save_df[, mrmrRanking]
  reorder_df$Labels <- labels
  write.csv(reorder_df, paste('D:/data/RNASeq/dataset/mRMR-LC-2classes-RNA-', mode ,SPLIT,'.csv', sep=""), row.names=TRUE)
}

mode = "test"

for (split in 7:10){
  create_dataset(split, mode)
}


