library(dplyr)

folder = "/gladstone/finkbeiner/linsley/TM_analysis/GXYTMP-20231207-2-MsN-lentieos/CSVS"


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
channel_names <- c('GFP-DMD1', 'GFP-DMD2', 'RFP1', 'RFP2')
channeldata <- filter(channeldata, channel %in% channel_names)
glimpse(channeldata)

# Check if you have the intensity per cell for your channel. If the channels are not present after the merge, then run the Intensity for your channel.
intensitycelldata_channeldata <- merge(intensitycelldata, channeldata, by='channeldata_id', suffixes=c("", ".dontuse"))
print(unique(intensitycelldata_channeldata$channel))

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