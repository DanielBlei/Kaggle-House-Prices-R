dir = "C:\\Users\\danie\\Desktop\\Comp\\House sales"

setwd(dir)
library(dplyr)
library(mice)
library(caret)
library(xgboost)
train <- read.csv("train.csv", stringsAsFactors = F)
test<- read.csv("test.csv", stringsAsFactors = F)
sample <- read.csv("sample_submission.csv")
model <- train$SalePrice

train <- train[-1]
test <- data.frame(SalePrice = rep(0, nrow(test)), test[,]) %>% within(rm('Id'))
df <- rbind2(train,test)

# Fxixng missing Values
df$PoolQC <- as.character(df$PoolQC)
df$PoolQC[is.na(df$PoolQC)] <- "None"
table(df$PoolQC) ; sort(unique(df$PoolQC))

# Basement Exposure
table(df$BsmtExposure) ; str(df$BsmtExposure)
df$BsmtExposure[is.na(df$BsmtExposure)] <- 'No'
any(is.na(df$BsmtExposure))

# MiscFfeature
df$MiscFeature <- as.character(df$MiscFeature)
df$MiscFeature[is.na(df$MiscFeature)] <- 'None'

# Fence
df$Fence <- as.character(df$Fence)
df$Fence[is.na(df$Fence)] <- "None"
table(df$Fence) ; sort(unique(df$Fence))

# Alley
df$Alley <- as.character(df$Alley)
df$Alley[is.na(df$Alley)] <- "None"
table(df$Alley)

# Fire Place Qu
df$FireplaceQu <- as.character(df$FireplaceQu)
df$FireplaceQu[is.na(df$FireplaceQu)] <- 'None'

# LotFrontage
summary(df$LotFrontage)
# replace the missing values with the median value  
df$LotFrontage[is.na(df$LotFrontage)] <- 68
any(is.na(df$LotFrontage))  # check if no missing values

# Garage Condition
table(df$GarageCond)
# replace with the most common occurence
df$GarageCond[is.na(df$GarageCond)] <- 'TA'
table(df$GarageCond)

# Garage Quality
table(df$GarageQual)
df$GarageQual[is.na(df$GarageQual)] <- 'TA'

# Garage Finish
df$GarageFinish <- as.character(df$GarageFinish)
df$GarageFinish[is.na(df$GarageFinish)] <- 'None'

# Garage Year built
summary(df$GarageYrBlt)

# create a loop to assign Garage year from Year built
for(i in 1:length(df$GarageYrBlt)){
  if(is.na(df$GarageYrBlt[i])){
    df$GarageYrBlt[i] <- df$YearBuilt[i]
  }else{
    df$GarageYrBlt[i] <- df$GarageYrBlt[i]
  }
}

# Garage Type
table(df$GarageType)
# replace with the most common occurence
df$GarageType[is.na(df$GarageType)] <- "Attchd"

# basement Condition
table(df$BsmtCond)
df$BsmtCond[is.na(df$BsmtCond)] <- 'TA'

# basement Quality
df$BsmtQual[is.na(df$BsmtQual)] <- 'TA'

# BsmtFinType2
table(df$BsmtFinType2)
# replace with the most common occurence
df$BsmtFinType2[is.na(df$BsmtFinType2)] <- 'Unf'

# BsmtFinType1
df$BsmtFinType1 <- as.character(df$BsmtFinType1)
df$BsmtFinType1[is.na(df$BsmtFinType1)] <- 'None' 

# massVnr Type
df$MasVnrType[is.na(df$MasVnrType)] <- 'None'

# massVnr Area
df$MasVnrArea[is.na(df$MasVnrArea)] <- 0

# MS zoning
df$MSZoning[is.na(df$MSZoning)] <- 'RL'


# Utilities
table(df$Utilities)
df$Utilities <- as.character(df$Utilities)
df$Utilities[is.na(df$Utilities)] <- "Allpub"

# BSmt FIns F1
df$BsmtFinSF1[is.na(df$BsmtFinSF1)] <- 0

# BSmt FIns F2
df$BsmtFinSF2[is.na(df$BsmtFinSF2)] <- 0

# Bsmt Unfs SF
df$BsmtUnfSF[is.na(df$BsmtUnfSF)] <- 0

# Total Bsmt SF
df$TotalBsmtSF[is.na(df$TotalBsmtSF)] <- 0


# Change all character variables to factor
for(i in colnames(df[,sapply(df, is.character)])){
  df[,i] <- as.factor(df[,i])
}

df$YrSold <- as.factor(df$YrSold)
df$MoSold <- as.factor(df$MoSold)
df$MSSubClass <- as.factor(df$MSSubClass)

# use mice to impute values
df_raw <- mice(df, m=3, method='cart', maxit=1)

df <- complete(df_raw)

# Check to vonfirm no missing values 
any(is.na(df))


# Split the data set

train0 <- df[1:1460,]
test0 <- df[1461:2919,]


train0[] <- lapply(train0, as.numeric)
test0[] <- lapply(test0, as.numeric)

dtrain=xgb.DMatrix(as.matrix(train0))
dtest=xgb.DMatrix(as.matrix(test0))


set.seed(233)

cv.ctrl <- trainControl(method = "repeatedcv",number = 20,repeats=5,
                        verboseIter=T,
                        classProbs= F)

xgb.grid <- expand.grid(nrounds = 800,
                        max_depth = c(5),
                        eta = c(0.015),
                        gamma = c(0.01),
                        colsample_bytree = 0.75,
                        min_child_weight = c(0),
                        subsample = c(0.5))

training <- train(log1p(SalePrice) ~
  ., data = train0,
  method="xgbTree",
  stratified=TRUE,
  verbose = 1, 
  eval_metric="rmse",
  trControl = cv.ctrl,
  tuneGrid = xgb.grid)


test0$SalePrice <-  expm1(predict(training,newdata= test0))
sample$SalePrice<- test0$SalePrice
write.csv(sample,"xgb_log.csv",row.names = FALSE)
