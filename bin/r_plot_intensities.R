library(dplyr)
library(ggplot2)
library(ggfortify)
library(survival)
library(dbplyr)
library(dplyr)
library(optparse)
# folder = "/gladstone/finkbeiner/linsley/TM_analysis/GXYTMP-20231207-2-MsN-lentieos/CSVS"
folder = "/gladstone/finkbeiner/linsley/TM_analysis/GXYTMP-20231207-2-MsN-minisog/CSVS"


experimentdata = read.csv(file.path(folder, "experimentdata.csv"))
welldata = read.csv(file.path(folder, "welldata.csv"))
tiledata = read.csv(file.path(folder, "tiledata.csv"))
celldata = read.csv(file.path(folder, "celldata.csv"))
channeldata = read.csv(file.path(folder, "channeldata.csv"))
dosagedata = read.csv(file.path(folder, "dosagedata.csv"))
intensitycelldata = read.csv(file.path(folder, "intensitycelldata.csv"))
# Rename id column to channeldata_id. The id may get replaced by another dataframe's id column. This avoids confusion. However, if you do not wish to rename columns, you can
# merge with df <- merge(df1, df2, by.x='column from df1', by.y'='column from df2')
colnames(channeldata)[colnames(channeldata) == "id"] = "channeldata_id"
colnames(welldata)[colnames(welldata) == "id"] = "welldata_id"
colnames(tiledata)[colnames(tiledata) == "id"] = "tiledata_id"
colnames(celldata)[colnames(celldata) == "id"] = "celldata_id"
colnames(channeldata)[colnames(channeldata) == "id"] = "channeldata_id"

#channel_names <- c('GFP-DMD1', 'RFP1', 'Cy5', 'Brightfield', 'BLUE-DMD-blocked', 'DAPI-DMD')
channel_names <- c('GFP-DMD1', 'RFP1', 'BLUE-DMD-blocked')
channeldata <- filter(channeldata, channel %in% channel_names)
glimpse(channeldata)

# Check if you have the intensity per cell for your channel. If the channels are not present after the merge, then run the Intensity for your channel.
intensitycelldata_channeldata <- merge(intensitycelldata, channeldata, by='channeldata_id', suffixes=c("", ".dontuse"))
print(unique(intensitycelldata_channeldata$channel))


# Merge to get intensities for celldata. include well, tile, dosage info. 
df1 <- merge(tiledata, welldata, by='welldata_id',suffixes=c("", ".dontuse"))
print(unique(df1$well))  # List wells
print(unique(df1$timepoint))  # List timepoints
print(unique(df1$condition))  # List conditions (control, disease, stim, unstim,)
df2 <- merge(df1, dosagedata, by='welldata_id', suffixes=c("", ".dontuse"))
glimpse(df2)
print(unique(df2$name))  # Name of the drug or treatment or biosensor
print(unique(df2$kind))  # Show the kind of drug or treatment or biosensor
df3 <- merge(celldata, df2, by='tiledata_id', suffixes=c("", ".dontuse"))
glimpse(df3)
print(unique(df3$stimulate))  # See if dataset has DMD Stimulate, Values: False and True
df4 <- merge(intensitycelldata, df3, by='celldata_id', suffixes=c("", ".dontuse"))
glimpse(df4)
df5 <- merge(df4, channeldata, by='channeldata_id', suffixes=c("", ".dontuse"))
data <- df5 %>% select(-contains("dontuse"))
print(unique(data$channel))

# link stimulation intensities from dmd to dataframe

stim1 <- merge(df2, channeldata, by='channeldata_id', suffixes=c("", ".dontuse"))
stim1 <- stim1%>% select(-contains("dontuse"))

# Check channels are there
print(unique(stim1$channel))

# Filter only stim channel
stim_channel_name <- c('BLUE-DMD-blocked')
stim2 <- filter(stim1, channel %in% stim_channel_name)

# Rename columns of interest to prevent conflict
colnames(stim2)[colnames(stim2) == "channel"] = "stim_channel"
colnames(stim2)[colnames(stim2) == "exposure"] = "stim_exposure"
colnames(stim2)[colnames(stim2) == "blue"] = "stim_blue"

# Identifier for tiles (tiledata_id identifies the stim channel only.)
stim2$tileidentifier=paste(stim2$well,stim2$tile,stim2$timepoint, sep="_")
data$tileidentifier=paste(data$well,data$tile,data$timepoint, sep="_")

stim3  = stim2[c("tileidentifier", "stim_channel", "stim_exposure", "stim_blue")]

datastim <- merge(data, stim3, by='tileidentifier', suffixes=c("", ".dontuse"))

###calculate GEDI ratio
# data$GEDIratio=data[data$channel=='RFP1',]$intensity_mean/data[data$channel=='GFP-DMD1',]$intensity_mean
# Calculate ratio
data <- data[!is.na(data$intensity_mean),]

data <- data %>% group_by(celldata_id) %>% filter(n() > 1)

data$track=paste(data$cellid,data$tile,data$well, sep="_")
data$timepoint=as.numeric(data$timepoint)
data$tile=as.numeric(data$tile)

dataT0 <- subset(data, data$channel=='GFP-DMD1' & data$timepoint==0 & data$intensity_mean > 2000)
keep_tracks = dataT0$track
data <- subset(data, data$track %in% keep_tracks)


### QC test single well
datawell=subset(data,data$well=="B3")
datawell=subset(datawell,datawell$tile==16)
# datawell=subset(datawell,datawell$area > 400)
datawellRFP=subset(datawell,datawell$channel=="RFP1")
datawell=subset(datawell,datawell$channel=="GFP-DMD1")
# datawellT0 <- subset(datawell, datawell$timepoint==0 & datawell$intensity_mean > 4000)
# datawell=subset(datawell,datawell$timepoint>3)
# keep_tracks = datawellT0$track
# datawell <- subset(datawell, datawell$track %in% keep_tracks)
# datawellRFP <- subset(datawellRFP, datawellRFP$track %in% keep_tracks)

##Well GFP Plot
ggplot(data=datawell, aes(y=intensity_mean, x=timepoint))+ 
#  ylim(0,5000)+ 
  geom_jitter(size=0.01)
   ##Well RFP Plot
ggplot(data=datawellRFP, aes(y=intensity_mean, x=timepoint))+ 
#  ylim(0,5000)+ 
  geom_jitter(size=0.01)


data2 <- data %>%
  group_by(celldata_id) %>%
  mutate(GEDIratio = if_else(channel == "GFP-DMD1", intensity_mean[channel == "RFP1"]/intensity_mean,  NA_real_)) %>%
  filter(channel == "GFP-DMD1")
print('Calculated Ratio')
dataRFP =subset(data,data$channel=="RFP1")
dataGFP =subset(data,data$channel=="GFP-DMD1")
dataGFP <- dataGFP[!is.na(dataGFP$intensity_mean),]
### QC plot areas
ggplot(data, aes(x=area)) +
  theme(legend.position="none")+
  geom_histogram(bins=900)+
  theme_classic()


### QC plot intensities
ggplot(data, aes(x=intensity_mean)) +
  theme(legend.position="none")+
  geom_histogram(binwidth = 900)+
  theme_classic()
##ALL data GFP plot
ggplot(data=dataGFP, aes(y=intensity_mean, x=timepoint))+ 
#  ylim(0,5000)+ 
  geom_jitter(size=0.01)





## Gedi Ratio single well
ggplot(data=datawell, aes(y=ratio, x=timepoint, color=stimulate))+ 
 ylim(0,0.1)+ 
  geom_jitter(size=0.1)

ggplot(data=datawell, aes(y=ratio, x=timepoint,group=track, color=track))+ 
  # ylim(0,0.5)+ 
  theme(legend.position="none")+
  geom_hline(yintercept=0.25, linetype="dashed", 
             color = "green", size=0.5)+
  # scale_colour_gradient(limits=c(0, 10000), low="red", high="blue")+
  geom_line()
