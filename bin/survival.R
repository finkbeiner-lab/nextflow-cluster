#!/usr/bin/env Rscript

library(survival)
library(ggplot2)
library(ggfortify)
library(optparse)
library(RPostgreSQL)

option_list = list(
  make_option(c("-exp", "--exp"), type="character", default=NULL, 
              help="Experiment name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="survival.csv", 
              help="output file name [default= %default]", metavar="character")
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
args = commandArgs(trailingOnly=TRUE)


drv <- dbDriver("PostgreSQL")
con <- d:Connect(drv, host="fb-postgres01.gladstone.internal", user="postgres", password="mysecretpassword", dbname="galaxy", port="5432")
rs <- dbSendQuery(con, "select * celldata"); 
df <- fetch(rs)

survcsv<-file.path(args[1], 'survival_data.csv')
if (!file.exists(survcsv)){
  errorCondition(paste0("File does not exist: ", survcsv))
}

# exp_dir <- "/gladstone/finkbeiner/elia/BiancaB/Imaging_Experiments/iMG_cocultures/GXYTMP/IMG-coculture-2-061522-Th3"
df <- read.csv(survcsv)
df$label <- as.factor(df$label)
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

