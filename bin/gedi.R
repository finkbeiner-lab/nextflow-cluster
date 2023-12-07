library(dplyr)
library(ggplot2)
library(ggfortify)
library(survival)
library(dbplyr)
library(dplyr)
library(optparse)
source("rsql.R")

option_list = list(
    make_option(c("--exp"), type="character", default='20231005-MS-10-minisog-IF', 
                help="Experiment name", metavar="character"),
        make_option(c("--channel1"), type="character", default='RFP1', 
    help="Channel 1, numerator", metavar="character"),
        make_option(c("--channel2"), type="character", default='GFP-DMD1', 
    help="Channel 2, denominator", metavar="character"),
        make_option(c("--well"), type="character", default='A3', 
    help="Well", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="survival.csv", 
                help="output file name [default= %default]", metavar="character")
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
args = commandArgs(trailingOnly=TRUE)

experiment = opt$exp
print(experiment)


db <- tbl(con, "experimentdata")

result <- db %>%
  inner_join(tbl(con, "channeldata"), by = c("id" = "experimentdata_id")) %>%
  inner_join(tbl(con, "welldata"), by = "id") %>%
  inner_join(tbl(con, "tiledata"), by = c("welldata_id" = "id")) %>%
  inner_join(tbl(con, "intensitycelldata"), by = c("id" = "channeldata_id")) %>%
  inner_join(tbl(con, "celldata"), by = c("id" = "celldata_id", "id" = "tiledata_id")) %>%
  inner_join(tbl(con, "dosagedata"), by = c("welldata_id" = "id")) %>%
  filter(
    experimentdata.experiment == experiemnt & 
    (welldata.well == opt$well | welldata.well == "all")
  ) %>%
  select(
    experimentdata.experiment, experimentdata.id, celldata.cellid, celldata.randomcellid,
    tiledata.tile, tiledata.timepoint, intensitycelldata.celldata_id,
    intensitycelldata.intensity_max, intensitycelldata.intensity_mean,
    channeldata.channel, welldata.well, dosagedata.name, celldata.stimulate
  ) %>%
  collect()

# query <- sprintf("SELECT DISTINCT experimentdata.experiment, experimentdata.id, celldata.cellid, 
# celldata.randomcellid, tiledata.tile, tiledata.timepoint,
# intensitycelldata.celldata_id, intensitycelldata.intensity_max, intensitycelldata.intensity_mean, 
# channeldata.channel, welldata.well, dosagedata.name, celldata.stimulate
#     from experimentdata
#     inner join channeldata
#         on channeldata.experimentdata_id = experimentdata.id
#     inner join welldata
#         on welldata.id=channeldata.welldata_id
#     inner join tiledata
#     on tiledata.welldata_id = welldata.id
#     inner join intensitycelldata
#         on intensitycelldata.channeldata_id=channeldata.id
#     inner join celldata
#         on celldata.id=intensitycelldata.celldata_id and celldata.tiledata_id=tiledata.id
#     inner join dosagedata
#     on dosagedata.welldata_id=welldata.id
#     where experimentdata.experiment=\'%s\' and welldata.well=\'%s\';"
#     ,experiment, opt$well)


print(query)
exp_row = get_row('experimentdata', sprintf("experiment=\'%s\'", opt$exp))
analysisdir = exp_row[1, 'analysisdir']
df <- get_df(query)

# group_counts <- data %>%
#   group_by(cellid, well, tile) %>%
#   filter(any(timepoint==0)) %>%
#   tally()

# df <- data %>%
#   group_by(cellid, well, tile) %>%
#   filter(any(timepoint==0)) %>%
#   ungroup()

# Create a jittered scatter plot
channel1_df <- df %>% filter(channel==opt$channel1)
channel2_df <- df %>% filter(channel==opt$channel2)

ggplot(channel1_df, aes(x=timepoint, y=intensity_mean)) + 
  geom_jitter(width=0.2) +  # The width argument controls the amount of jittering on the x-axis
  labs(title=paste0(opt$channel1, ' ', opt$exp), x="Timepoint", y=opt$channel1) 


ggplot(channel2_df, aes(x=timepoint, y=intensity_mean)) + 
  geom_jitter(width=0.2) +  # The width argument controls the amount of jittering on the x-axis
  labs(title=paste0(opt$channel2, ' ', opt$exp), x="Timepoint", y=opt$channel2) 

# Calculate ratio
df <- df %>%
  group_by(celldata_id) %>%
  mutate(ratio = if_else(channel == opt$channel2, intensity_mean[channel == opt$channel1]/intensity_mean,  NA_real_)) %>%
  filter(channel == opt$channel2)
print('Calculated Ratio')
# drop nan
# df <- df %>%
#   filter(!is.na(ratio))
# print('Filtered Nan')
# Threshold gedi ratio

ggplot(df, aes(x=timepoint, y=ratio)) + 
  geom_jitter(width=0.2) +  # The width argument controls the amount of jittering on the x-axis

  labs(title=paste0('ratio', ' ', opt$exp), x="Timepoint", y='ratio') 

gedi_threshold=.1
df <- df %>%
  mutate(is_dead = ratio > gedi_threshold)
print('Thresholded gedi ratio.')
# Reduce tracks to summarise time of death
first_group <- df %>%
  group_by(cellid, well, tile) %>%
  filter(n() > 1) %>% 
  mutate(group_id = cur_group_id()) %>%
  filter(group_id == 1) %>%
  ungroup() %>%
  select(-group_id)

survival_df_tst <- first_group %>%
  group_by(cellid, well, tile) %>%
  filter(n() > 1) %>% 
  arrange(timepoint) %>%
  filter(if (any(is_dead == TRUE)) is_dead == TRUE else row_number() == n()) %>%
  slice(1) %>%
    ungroup()

survival_df <- df %>%
  group_by(cellid, well, tile) %>%
  filter(n() > 1) %>% 
  arrange(timepoint) %>%
  filter(if (any(is_dead == TRUE)) is_dead == TRUE else row_number() == n()) %>%
  slice(1) %>%
    ungroup()
print('Reduced tracks')
# exp_dir <- "/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/GXYTMP/IMG-coculture-2-061522-Th3"
# survival_df$is_dead <- as.factor(survival_df$is_dead)
# survival_df$stimulate <- as.factor(survival_df$stimulate)

survival_df.KM<-survfit(Surv(timepoint, is_dead) ~ stimulate, data=survival_df, type="kaplan-meier")
png(file=file.path(analysisdir, "kaplanmeier.png"), width=800, height=800)
print(paste0("Saved kaplanmeier.png to ", analysisdir))
autoplot(survival_df.KM)

plot(survival_df.KM, col=c(2,4,6), xlab="Days", ylab="Survival", main=sprintf("Kaplan Meier %s", opt$exp))
dev.off()



cox = coxph(Surv(timepoint, is_dead) ~ stimulate, data=survival_df)
summary(cox)
savepath <- file.path(analysisdir, "coxph.csv")
summ<-summary(cox)
class(summ$coefficients)
write.csv(summ$coefficients, savepath)
print(paste0("Saved coxph.csv to ", analysisdir))

