#!/usr/bin/env Rscript

library(survival)
library(ggplot2)
library(ggfortify)
library(optparse)
source("rsql.R")

option_list = list(
  make_option(c("-exp", "--exp"), type="character", default=NULL, 
              help="Experiment name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="survival.csv", 
              help="output file name [default= %default]", metavar="character")
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
args = commandArgs(trailingOnly=TRUE)

experiment = opt$exp

query <- sprintf("SELECT DISTINCT experimentdata.experiment, experimentdata.id, channeldata.channel, 
welldata.well, dosagedata.name, celldata.stimulate, modelcropdata.prediction, modelcropdata.groundtruth, modelcropdata.stage
    from experimentdata
    inner join channeldata
        on channeldata.experimentdata_id = experimentdata.id
    inner join welldata
        on welldata.id=channeldata.welldata_id
    inner join tiledata
    on tiledata.welldata_id = welldata.id
    inner join celldata
        on celldata.id=intensitycelldata.celldata_id
    inner join modelcropdata
        on modelcropdata.celldata_id=celldata.id
    inner join dosagedata
    on dosagedata.welldata_id=welldata.id
    where experimentdata.experiment=%s and welldata.well=%s and tiledata.timepoint=%d;",experiment, well, timepoint)

df <- get_df(query)

# exp_dir <- "/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/GXYTMP/IMG-coculture-2-061522-Th3"
df$label <- as.factor(df$groundtruth)
df$condition <- as.factor(df$condition)

df.KM<-survfit(Surv(time_of_death, is_dead) ~ condition, data=df, type="kaplan-meier")
png(file=file.path(args[1], "km_R.png"), width=800, height=800)

autoplot(df.KM)
dev.off()

# plot(df.KM, col=c(2,4,6), xlab="Days", ylab="Survival", main="Kaplan Meier")

cox = coxph(Surv(time_of_death, is_dead) ~ condition, data=df)
summary(cox)
savepath <- file.path(args[1], "coxph.csv")
summ<-summary(cox)
class(summ$coefficients)
write.csv(summ$coefficients, savepath)
print(paste0("Saved coxph.csv to ", args[1]))

