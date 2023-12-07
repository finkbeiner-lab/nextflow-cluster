library(dplyr)

folder = "/gladstone/finkbeiner/linsley/Kaushik/GXYTMPS/GXYTMP-20231109-1-MsN-cry2tdp43-updated/CSVS"


experimentdata = read.csv(file.path(folder, "experimentdata.csv"))
welldata = read.csv(file.path(folder, "welldata.csv"))
tiledata = read.csv(file.path(folder, "tiledata.csv"))
celldata = read.csv(file.path(folder, "celldata.csv"))
channeldata = read.csv(file.path(folder, "channeldata.csv"))
dosagedata = read.csv(file.path(folder, "dosagedata.csv"))
intensitycelldata = read.csv(file.path(folder, "intensitycelldata.csv"))
colnames(channeldata)[colnames(channeldata) == "id"] = "channeldata_id"

channel_names <- c('GFP-DMD1', 'RFP1', 'Cy5', 'Brightfield', 'BLUE-DMD-blocked', 'DAPI-DMD')
channeldata <- filter(channeldata, channel %in% channel_names)
glimpse(channeldata)

intensitycelldata_channeldata <- merge(intensitycelldata, channeldata, by='channeldata_id', suffixes=c("", ".dontuse"))
print(unique(intensitycelldata_channeldata$channel))
df1 <- merge(tiledata, welldata, by.x='welldata_id', by.y='id',suffixes=c("", ".dontuse"))



df2 <- merge(celldata, df1, by.x='tiledata_id', by.y='id', suffixes=c("", ".dontuse"))
glimpse(df2)
df3 <- merge(intensitycelldata, df2, by.x='celldata_id', by.y='id', suffixes=c("", ".dontuse"))
glimpse(df3)
df4 <- merge(df3, channeldata, by.x='channeldata_id', by.y='id', suffixes=c("", ".dontuse"))
data <- df4 %>% select(-contains("dontuse"))


