#' @title MRID functions
#'
#' @description This functions makes MRID ratios and plots
#' 
#' @author Julia Kaye
#' @author Thomas Reuben
#' @author Stephanie Lam
#' 
#' @import ggplot2
#' @import dplyr
#' @import plyr
#' @import gridExtra
#'
#' @param data cell_data.csv
#' @param input_data modified cell_data dataframe
#' @param output_path output directory
#' @param qc_path path with manual curation of live/dead cell csv
#' @param qc TRUE/FALSE flag
#' @param report_missing print out message of wells having no data
#' @param timepoint_removal list of timepoint to be removed from cell_data.csv
#' @param well_removal list of wells to be removed from cell_data.csv
#' @param plate_layout csv contains Sci_SampleID, Sci_WellID, Drug
#' @param user_choice_of_sd # of standard deviation away from median
#' @param user_threshold user threshold for live and dead ratio
#' @param ylim_max y-axis range in plots
#' @param user_well_outlier_method could be either "mean" "median" or "none", 
#' remove wells with mean/median that's greater than average mean/median of the entire plate
#' @param user_live_threshold final threshold set for live cells
#' @param user_dead_threshold final threshold set for dead cells
#' @param cell_red_flag flag for having less than a number of live cells in heatmap
#' @param calc_z_score TRUE/FALSE flag for calculate gedi z scores, normalize by well per timepoint
#' 
#' @return 
#'
#' @export
#'
#' @examples ratio_df = mrid_ratio(cell_data ,timepoint_removal, well_removal, plate_layout, output_path)
#' @examples manual_curation_qc(curation_path, qc)
#' @examples mrid_sd_plot(ratio_df, user_choice_of_sd=2, user_threshold=0.08, ylim_max=0.35)
#' @examples mrid_violin_plot(ratio_df)
#' @examples mrid_condition_plot(ratio_df, user_threshold=0.08, ylim_max=0.35)
#' @examples final_df = mrid_final_plot(ratio_df, output_path, user_well_outlier_method=none, user_live_threshold=0.08, user_dead_threshold=0.1, ylim_max=0.6)
#' @examples livedead = mrid_livedead_csv(final_df, output_path, expt_name='UC-PCMI7", user_live_threshold=0.08, user_dead_threshold=0.1)
#' @examples mrid_heatmap_plot(livedead, output_path,cell_red_flag=30)
#' 

create_output_dir <- function(output_path){
  #generate output folder
  if (!file.exists(output_path)){
    dir.create(output_path, showWarnings = FALSE)
  }
  #setwd(output_path)
}

mrid_manual_curation_qc <- function(qc_path, qc){
  #read qc csv file if manually curation is done
  if(qc){
    curation <- read.csv(qc_path)
    curation['Sci_WellID'] <- unlist(lapply(1:dim(curation)[1], function(x) strsplit(as.character(curation$Fname[x]),'_')[[1]][5]))
    curation['Timepoint'] <- unlist(lapply(1:dim(curation)[1], function(x) strsplit(as.character(curation$Fname[x]),'_')[[1]][3]))
    curation$Timepoint <- gsub('T','',curation$Timepoint)
    curation['ObjectLabelsFound'] <- unlist(lapply(1:dim(curation)[1], function(x) strsplit(as.character(curation$Fname[x]),'_')[[1]][13]))
    curation['ObjectLabelsFound'] <- gsub('.tif','',curation$ObjectLabelsFound)
    curation <- curation[, c('Fname', 'Column', 'Sci_WellID', 'Timepoint', 'ObjectLabelsFound')]
    
    #merge curation and ratio dataframe
    curation_ratio <- merge(ratio, curation, all.y = TRUE, by=c('Sci_WellID', 'Timepoint', 'ObjectLabelsFound'))
    
    #filtered images labelled as x or X
    curation_ratio_filter <- curation_ratio[grepl('[x|X]',curation_ratio$Column),]
    print(paste0("Number of filtered images: ", dim(curation_ratio_filter)[1]))
    print(summary(curation_ratio_filter$ratio))
    write.csv(curation_ratio_filter, file.path(outputdir, "filteredimages_ratio.csv"))
    
    #live images labelled as 1
    curation_ratio_live <- curation_ratio[grepl('1',curation_ratio$Column),]
    print(paste0("Number of live images: ", dim(curation_ratio_live)[1]))
    print(summary(curation_ratio_live$ratio))
    write.csv(curation_ratio_live, file.path(outputdir, "liveimages_ratio.csv"))
    
    #dead images labelled as 0
    curation_ratio_dead <- curation_ratio[grepl('0',curation_ratio$Column),]
    print(paste0("Number of dead images: ", dim(curation_ratio_dead)[1]))
    print(summary(curation_ratio_dead$ratio))
    write.csv(curation_ratio_dead, file.path(outputdir, "deadimages_ratio.csv"))
  }
}

cell_count <- function(data, report_missing){
  #make sure there are only one channel
  data = data[data$MeasurementTag == unique(data$MeasurementTag)[1],]
  
  #count by condition
  if("Drug" %in% colnames(data)){
    condition_count = data %>% group_by(Sci_SampleID, Drug, Timepoint) %>% dplyr::summarise(n=n())
  }else{
    condition_count = data %>% group_by(Sci_SampleID, Timepoint) %>% dplyr::summarise(n=n())
  }
  write.csv(condition_count, "cell_count.csv")
  
  #count by well
  well_count = data %>% group_by(Sci_WellID,Timepoint) %>% dplyr::summarise(n=n())
  well_count_all = NULL
  for(tp in 1:length(levels(as.factor(well_count$Timepoint)))){
    well_count_tp = NULL
    well_count_tp = well_count[well_count$Timepoint == levels(as.factor(well_count$Timepoint))[tp],]
    
    missing = unique(well_count$Sci_WellID)[!unique(well_count$Sci_WellID) %in% well_count_tp$Sci_WellID]
    if(report_missing){
      if(length(missing >0)){
        print(paste0(paste(missing,collapse = ','), " does not have any cells at T",levels(as.factor(well_count$Timepoint))[tp]))
      }
    }
    if(length(missing) > 0){
      missing_df <- data.frame(Sci_WellID=missing,Timepoint=levels(as.factor(well_count$Timepoint))[tp], n=rep(NA,length(missing)))
      well_count_tp = plyr::rbind.fill(well_count_tp, missing_df)
    }
    well_count_all = plyr::rbind.fill(well_count_all,well_count_tp)
  }
  well_count_sorted = well_count_all[order(well_count_all$Timepoint,well_count_all$Sci_WellID),]
  write.csv(well_count_sorted, "cell_count_well.csv")
  
  return(list(condition_count,well_count_sorted))
}

deduplicate_by_small_deletion <- function(data) {
  # Deduplication strategy: Takes cell_data and removes the smaller object duplicates, returns de-duplicated data
  print(paste('Initiated deduplication with (number of rows):', nrow(data)))
  data["PlateWellObj"] <- factor(paste(data$Sci_PlateID, data$Sci_WellID, 
                                       data$ObjectLabelsFound, data$Timepoint, data$MeasurementTag, sep="_"))
  data = data[order(data[,'PlateWellObj'],-data[,'BlobArea']),]
  data = data[!duplicated(data$PlateWellObj),]
  data = data[-grep("PlateWellObj", colnames(data))] # remove the ID column before writing
  data = data[order(data[,'Sci_PlateID'], data[,'Sci_WellID'],data[,'Timepoint'], data[,'ObjectLabelsFound']),]
  print(paste('Completed deduplication with (number of rows):', nrow(data)))
  return(data)
}

mrid_ratio <- function(data, timepoint_removal, well_removal, plate_layout, output_path, calc_z_score = FALSE){
  #function to calculate mrid ratio, expect data with FITC and RFP channel
  #create output dir
  create_output_dir(output_path)
  
  #filter timepoints and duplications
  #To test: data = cell_data
  data = data[!data$Timepoint%in% timepoint_removal, ]
  data = deduplicate_by_small_deletion(data)
  
  data$timepoint_well_neuron=paste(data$Timepoint,data$Sci_WellID,data$ObjectLabelsFound,sep="_")
  
  # Define a function to expand well ranges
  expand_well_range <- function(well_range) {
    parts <- unlist(strsplit(well_range, "-"))
    if (length(parts) == 2) {
      well_start <- parts[1]
      well_end <- parts[2]
      
      # Extract the alphabetic and numeric parts of well names
      alpha_start <- gsub("[0-9]+", "", well_start)
      num_start <- as.character(gsub("[A-Z]+", "", well_start))
      alpha_end <- gsub("[0-9]+", "", well_end)
      num_end <- as.character(gsub("[A-Z]+", "", well_end))
      
      # Expand the range with leading zeros preserved
      well_range_expanded <- paste0(
        alpha_start,
        formatC(num_start:num_end, width = nchar(num_start), format = "d", flag = "0")
      )
      
      return(well_range_expanded)
    } else {
      return(well_range)
    }
  }
  # Apply the function to each element in well_removal
  expanded_wells <- unlist(lapply(well_removal, expand_well_range))
  expanded_wells
  
  # Remove the specified wells
  data <- data[!data$Sci_WellID %in% expanded_wells, ]
  
  #check measurementag
  if(any(grepl("FITC|GFP", unique(data$MeasurementTag))) && any(grepl("RFP", unique(data$MeasurementTag)))){
    #subset RFP and FITC
    FITC<-subset(data,MeasurementTag== unique(data$MeasurementTag)[grepl("FITC|GFP",unique(data$MeasurementTag))])
    RFP<- subset(data,MeasurementTag== unique(data$MeasurementTag)[grepl("RFP",unique(data$MeasurementTag))])
  
    #merge with platelayout
    RFP <- merge(RFP,plate_layout)
  }else{
    stop("There's no FITC/GFP or RFP channel")
  }
  
  #find cell counts per condition per timepoint
  count = cell_count(RFP, report_missing =TRUE)
  count$Drug = as.factor(count$Drug)
  
  #plot cell counts for general overview
  ggplot(data=count[[1]],aes(x=Timepoint,y=n,colour=Drug))+
    geom_line(aes(color=Drug))+
    geom_point(aes(color=Drug))+
    facet_wrap(~as.factor(Sci_SampleID))
  ggsave("cell_count.pdf", width = 10, height = 10)
  
  #remove wells not in T1
  well_count = count[[2]]
  wells_t1_missing = well_count[is.na(well_count$n) & well_count$Timepoint == levels(as.factor(well_count$Timepoint))[1],]$Sci_WellID
  
  #renaming GFP PixelIntensityMean to GFPPixelIntensityMean
  colnames(FITC)[colnames(FITC)=='PixelIntensityMean']<-"GFPPixelIntensityMean"
  
  if(calc_z_score){
    #Calculate Z-GEDI, normalize per well per timepoint
    Z_FITC = FITC %>% 
      group_by(Timepoint, Sci_WellID) %>%
      mutate(z_score = (GFPPixelIntensityMean - mean(GFPPixelIntensityMean)) / sd(GFPPixelIntensityMean))
    FITC_simple<-Z_FITC[,c("timepoint_well_neuron","GFPPixelIntensityMean","z_score")]
  }else{
    FITC_simple<-FITC[,c("timepoint_well_neuron","GFPPixelIntensityMean")]
  }
  
  #merge RFP pixel intensities with FITC pixel intensity information
  data<-merge(FITC_simple,RFP,by="timepoint_well_neuron")
  
  #Calculating GEDI ratio
  data$ratio<-((data$PixelIntensityMean/data$GFPPixelIntensityMean))
  if(calc_z_score) {data$z_ratio<-((data$PixelIntensityMean/data$z_score))}
  
  data <- merge(data, plate_layout)
  
  #remove wells without any cell at t1
  data <- data[!data$Sci_WellID %in% wells_t1_missing,]
  data$Sci_WellID <- factor(as.character(data$Sci_WellID))
  
  #output cell_data with gedi ratio column
  write.csv(data,"mrid_celldata_ratio.csv")
    
  return(data)
}

mrid_filter <- function(input_data){
  
  #calculate mean and median, filter outlier wells using T1
  med_summary = dplyr::group_by(input_data, Timepoint, Sci_WellID)
  med_summary = dplyr::summarise(med_summary, med = median(ratio),
                                 mean = mean(ratio))
  med_summary = med_summary[med_summary$Timepoint ==  levels(as.factor(input_data$Timepoint))[1],]
  t1_tmp = input_data[input_data$Timepoint == levels(as.factor(input_data$Timepoint))[1],]
  med_summary['med_filter'] = ifelse(med_summary$med > median(t1_tmp$ratio),'*',"")
  med_summary['mean_filter'] = ifelse(med_summary$mean > mean(t1_tmp$ratio),'*',"")
  
  #remove wells with mean higher than avg mean
  wells_remove_mean <- unique(med_summary[med_summary$mean_filter == '*',]$Sci_WellID)
  data_mean = input_data[!input_data$Sci_WellID %in% wells_remove_mean,]
  
  #remove wells with median higher than avg median
  wells_remove_med <- unique(med_summary[med_summary$med_filter == '*',]$Sci_WellID)
  data_med = input_data[!input_data$Sci_WellID %in% wells_remove_med,]
  
  return(list(med_summary, data_med, data_mean))
}

mrid_sd_plot <- function(input_data, user_choice_of_sd, user_threshold, ylim_max){
  #function to calculate and make ratio plot with different levels of sd
  
  med_mean_list = mrid_filter(input_data)
  med_summary = med_mean_list[[1]]
  data_med = med_mean_list[[2]]
  data_mean = med_mean_list[[3]]
  
  date <- Sys.Date()
  
  if(length(levels(input_data$Sci_WellID)) > 96){
    width_manual = 25
  }else{
    width_manual = 15
  }
  
  pdf(paste0("mrid_ratio_sd_",date,".pdf"),width = width_manual,height = 8)
  input_data$Sci_WellID <- as.factor(as.character(input_data$Sci_WellID))
  for(i in 1:length(levels(as.factor(input_data$Timepoint)))){
    tmp_sd <- input_data[input_data$Timepoint == levels(as.factor(input_data$Timepoint))[i],]
    
    SD = round(sd(tmp_sd$ratio, na.rm = TRUE),2)
    
    print(ggplot(data=tmp_sd ,aes(x=Sci_WellID,y=ratio))+
            geom_jitter(size=1, alpha=1/5, aes(colour=factor(Sci_SampleID)))+
            #change the threshold line
            geom_hline(yintercept = user_threshold)+
            #line for median
            geom_hline(aes(yintercept = median(ratio), group = Sci_WellID),colour = 'red',linetype = "solid")+
            #line for mean
            geom_hline(aes(yintercept = mean(ratio), group = Sci_WellID),colour = 'pink',linetype = "solid")+
            #line for median + sd
            geom_hline(aes(yintercept = median(ratio)+0.5*SD, group = Sci_WellID), colour = "pink", linetype = "dotted")+
            #line for median + 1sd
            geom_hline(aes(yintercept = median(ratio)+1*SD, group = Sci_WellID), colour = "pink", linetype = "dotted")+
            #User choice of sd
            geom_hline(aes(yintercept = median(ratio)+user_choice_of_sd*SD, group = Sci_WellID), colour = "blue", linetype = "dashed")+
            geom_text(data= med_summary, aes(x=Sci_WellID, y=med, label = med_filter), nudge_y = 0.05, colour = "red")+
            geom_text(data= med_summary, aes(x=Sci_WellID, y=mean, label = mean_filter), nudge_y = 0.05, colour = "black")+
            guides(colour = guide_legend(override.aes = list(size=5)))+
            #change the y-intercept based on user
            ylim(0, ylim_max)+
            theme(panel.grid.major = element_blank(),
                  legend.position = "none",
                  panel.grid.minor = element_blank(),
                  strip.background = element_blank(),
                  panel.border = element_blank(),
                  panel.background = element_rect(fill='white'),
                  axis.text.x = element_text(angle = 90))+
            ggtitle(label = paste0("Ratio for timepoint ", 
                                   levels(as.factor(ratio_df$Timepoint))[i]),
                    subtitle = paste0(" \nMedian(solid red) : ", round(median(tmp_sd$ratio),2), 
                                      " \nMean(solid pink) : ",round(mean(tmp_sd$ratio),2), 
                                      " \nMedian+0.5SD (dotted pink) : ",round(median(tmp_sd$ratio)+0.5*SD,2), 
                                      " \nMedian+1SD (dotted pink) : ",round(median(tmp_sd$ratio)+1*SD,2),
                                      " \nuser_choice_of_sd(blue): ",round(median(tmp_sd$ratio)+user_choice_of_sd*SD,2),
                                      " \nuser_threshold (black): ",user_threshold))
          + annotate("text",  x=Inf, y = Inf, label = "red * = well median > avg median \n black * = well mean > avg mean", vjust=1, hjust=1, color='red')
    )
    
  }
  dev.off()
  
  if(length(levels(factor(input_data$Drug))) > 1){
    pdf(paste0("mrid_ratio_sd_drug_",date,".pdf"),width = width_manual,height = 8)
    input_data$Sci_WellID <- as.factor(as.character(input_data$Sci_WellID))
    for(i in 1:length(levels(as.factor(input_data$Timepoint)))){
      tmp_sd <- input_data[input_data$Timepoint == levels(as.factor(input_data$Timepoint))[i],]
      tmp_sd['condition'] <- paste(tmp_sd$Sci_SampleID,tmp_sd$Drug,sep = '_')
      SD = round(sd(tmp_sd$ratio, na.rm = TRUE),2)
      
      print(ggplot(data=tmp_sd ,aes(x=Sci_WellID,y=ratio))+
              geom_jitter(size=1, alpha=1/5, aes(colour=factor(condition)))+
              #change the threshold line
              geom_hline(yintercept = user_threshold)+
              #line for median
              geom_hline(aes(yintercept = median(ratio), group = Sci_WellID),colour = 'red',linetype = "solid")+
              #line for mean
              geom_hline(aes(yintercept = mean(ratio), group = Sci_WellID),colour = 'pink',linetype = "solid")+
              #line for median + sd
              geom_hline(aes(yintercept = median(ratio)+0.5*SD, group = Sci_WellID), colour = "pink", linetype = "dotted")+
              #line for median + 1sd
              geom_hline(aes(yintercept = median(ratio)+1*SD, group = Sci_WellID), colour = "pink", linetype = "dotted")+
              #User choice of sd
              geom_hline(aes(yintercept = median(ratio)+user_choice_of_sd*SD, group = Sci_WellID), colour = "blue", linetype = "dashed")+
              geom_text(data= med_summary, aes(x=Sci_WellID, y=med, label = med_filter), nudge_y = 0.05, colour = "red")+
              geom_text(data= med_summary, aes(x=Sci_WellID, y=mean, label = mean_filter), nudge_y = 0.05, colour = "black")+
              guides(colour = guide_legend(override.aes = list(size=5)))+
              #change the y-intercept based on user
              ylim(0, ylim_max)+
              theme(panel.grid.major = element_blank(),
                    panel.grid.minor = element_blank(),
                    strip.background = element_blank(),
                    panel.border = element_blank(),
                    panel.background = element_rect(fill='white'),
                    axis.text.x = element_text(angle = 90))+
              ggtitle(label = paste0("Ratio for timepoint ", 
                                     levels(as.factor(ratio_df$Timepoint))[i]),
                      subtitle = paste0(" \nMedian(solid red) : ", round(median(tmp_sd$ratio),2), 
                                        " \nMean(solid pink) : ",round(mean(tmp_sd$ratio),2), 
                                        " \nMedian+0.5SD (dotted pink) : ",round(median(tmp_sd$ratio)+0.5*SD,2), 
                                        " \nMedian+1SD (dotted pink) : ",round(median(tmp_sd$ratio)+1*SD,2),
                                        " \nuser_choice_of_sd(blue): ",round(median(tmp_sd$ratio)+user_choice_of_sd*SD,2),
                                        " \nuser_threshold (black): ",user_threshold))
            + annotate("text",  x=Inf, y = Inf, label = "red * = well median > avg median \n black * = well mean > avg mean", vjust=1, hjust=1, color='red')
      )
      
    }
    dev.off()
  }
  
  #updated pdf
  pdf(paste0("mrid_ratio_outliers_by_mean_",date,".pdf"),width = width_manual, height = 8)
  for(i in 1:length(levels(as.factor(data_mean$Timepoint)))){
    tmp <- data_mean[data_mean$Timepoint == levels(as.factor(input_data$Timepoint))[i],]
    
    SD = round(sd(tmp$ratio, na.rm = TRUE),2)
    
    print(ggplot(data=tmp ,aes(x=Sci_WellID,y=ratio))+
            geom_jitter(size=1, alpha=1/5, aes(colour=factor(Sci_WellID)))+
            #change the threshold line
            geom_hline(yintercept = user_threshold)+
            #line for median
            geom_hline(aes(yintercept = median(ratio), group = Sci_WellID),colour = 'red',linetype = "solid")+
            #line for mean
            geom_hline(aes(yintercept = mean(ratio), group = Sci_WellID),colour = 'pink',linetype = "solid")+
            #line for median + sd
            geom_hline(aes(yintercept = median(ratio)+0.5*SD, group = Sci_WellID), colour = "pink", linetype = "dotted")+
            #line for median + 1sd
            geom_hline(aes(yintercept = median(ratio)+1*SD, group = Sci_WellID), colour = "pink", linetype = "dotted")+
            #User choice of sd
            geom_hline(aes(yintercept = median(ratio)+user_choice_of_sd*SD, group = Sci_WellID), colour = "blue", linetype = "dashed")+
            guides(colour = guide_legend(override.aes = list(size=5)))+
            #change the y-intercept based on user
            ylim(0, ylim_max)+
            theme(panel.grid.major = element_blank(),
                  legend.position = "none",
                  panel.grid.minor = element_blank(),
                  strip.background = element_blank(),
                  panel.border = element_blank(),
                  panel.background = element_rect(fill='white'),
                  axis.text.x = element_text(angle = 90))+
            ggtitle(paste0("Outliers wells removed, ratio for timepoint ", 
                           levels(as.factor(ratio_df$Timepoint))[i], 
                           " \nMedian(solid red) : ", round(median(tmp$ratio),2), 
                           " \nMean(solid pink) : ",round(mean(tmp$ratio),2), 
                           " \nMedian+0.5SD (dotted pink) : ",round(median(tmp$ratio)+0.5*SD,2), 
                           " \nMedian+1SD (dotted pink) : ",round(median(tmp$ratio)+1*SD,2),
                           " \nuser_choice_of_sd(blue): ",round(median(tmp$ratio)+user_choice_of_sd*SD,2),
                           " \nuser_threshold (black): ",user_threshold))
          
    )
  }
  dev.off()
  
  pdf(paste0("mrid_ratio_outliers_by_med_",date,".pdf"),width = width_manual, height = 8)
  for(i in 1:length(levels(as.factor(data_med$Timepoint)))){
    tmp <- data_med[data_med$Timepoint == levels(as.factor(input_data$Timepoint))[i],]
    
    SD = round(sd(tmp$ratio, na.rm = TRUE),2)
    
    print(ggplot(data=tmp,aes(x=Sci_WellID,y=ratio))+
            geom_jitter(size=1, alpha=1/5, aes(colour=factor(Sci_WellID)))+
            #change the threshold line
            geom_hline(yintercept = user_threshold)+
            #line for median
            geom_hline(aes(yintercept = median(ratio), group = Sci_WellID),colour = 'red',linetype = "solid")+
            #line for mean
            geom_hline(aes(yintercept = mean(ratio), group = Sci_WellID),colour = 'pink',linetype = "solid")+
            #line for median + sd
            geom_hline(aes(yintercept = median(ratio)+0.5*SD, group = Sci_WellID), colour = "pink", linetype = "dotted")+
            #line for median + 1sd
            geom_hline(aes(yintercept = median(ratio)+1*SD, group = Sci_WellID), colour = "pink", linetype = "dotted")+
            #User choice of sd
            geom_hline(aes(yintercept = median(ratio)+user_choice_of_sd*SD, group = Sci_WellID), colour = "blue", linetype = "dashed")+
            guides(colour = guide_legend(override.aes = list(size=5)))+
            #change the y-intercept based on user
            ylim(0, ylim_max)+
            theme(panel.grid.major = element_blank(),
                  legend.position = "none",
                  panel.grid.minor = element_blank(),
                  strip.background = element_blank(),
                  panel.border = element_blank(),
                  panel.background = element_rect(fill='white'),
                  axis.text.x = element_text(angle = 90))+
            ggtitle(paste0("Outliers wells removed, ratio for timepoint ", 
                           levels(as.factor(ratio_df$Timepoint))[i], 
                           " \nMedian(solid red) : ", round(median(tmp$ratio),2), 
                           " \nMean(solid pink) : ",round(mean(tmp$ratio),2), 
                           " \nMedian+0.5SD (dotted pink) : ",round(median(tmp$ratio)+0.5*SD,2), 
                           " \nMedian+1SD (dotted pink) : ",round(median(tmp$ratio)+1*SD,2),
                           " \nuser_choice_of_sd(blue): ",round(median(tmp$ratio)+user_choice_of_sd*SD,2),
                           " \nuser_threshold (black): ",user_threshold))
          
    )
  }
  dev.off()
  
}

mrid_violin_plot <- function(input_data){
  #violin plots(combine/per cellline)
  
  ggplot(data=input_data,aes(y=ratio))+
    geom_half_violin(side="r",fill="black",linewidth=0.5,position = position_dodge(width = 10))+ylim(0,1)+
    theme_classic()+labs( x ="Density")+scale_x_continuous(0, 1)
  ggsave("mrid_violin_plot_combine.pdf")
  
  ggplot(data=input_data[input_data$Timepoint == min(input_data$Timepoint),],aes(x=Drug, y=ratio, fill=Drug))+
    geom_half_violin()+
    theme_classic()+
    theme(axis.text.x = element_text(angle=45))+
    #change the y-intercept
    ylim(0,0.6)+
    facet_grid(~Sci_SampleID)
  ggsave("mrid_violin_plot_celllines.pdf")
}

mrid_condition_plot <- function(input_data, user_threshold, ylim_max){
  #function to make plots based on condition(Sci_SampleID/Drug)
  
  pdf("mrid_condition_plot.pdf", paper='A4r')
  for(i in 1:length(levels(as.factor(input_data$Drug)))){
    tmp <- input_data[input_data$Drug == levels(as.factor(input_data$Drug))[i],]
    print(ggplot(data=tmp,aes(x=Timepoint,y=ratio))+
            geom_jitter(size=1, alpha=1/5, aes(colour=factor(Timepoint)))+
            geom_hline(yintercept = user_threshold, colour = "red")+
            theme(legend.key = element_blank()) +
            guides(colour = guide_legend(override.aes = list(size=5)))+
            #change the y-intercept if needed
            ylim(0,ylim_max)+
            theme(panel.grid.major = element_blank(),
                  legend.position = "none",
                  panel.grid.minor = element_blank(),
                  strip.background = element_blank(),
                  panel.border = element_blank(),
                  panel.background = element_rect(fill='white'),
                  plot.subtitle = element_text(hjust = 1),
                  axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
            ggtitle(label = paste0("Drug ",levels(as.factor(ratio_df$Drug))[i]),
                    subtitle = paste0("red = user threshold ",user_threshold))+
            facet_wrap(.~Sci_SampleID))
          
            }
  dev.off()
  
  
  pdf("mrid_condition_plot_timepoint.pdf", paper='A4r')
  for(i in 1:length(levels(as.factor(input_data$Timepoint)))){
    tmp <- input_data[input_data$Timepoint == levels(as.factor(input_data$Timepoint))[i],]
    print(ggplot(data=tmp,aes(x=Drug,y=ratio))+
            geom_jitter(size=1, alpha=1/5, aes(colour=factor(Drug)))+
            geom_hline(yintercept = user_threshold, colour = "red")+
            guides(colour = guide_legend(override.aes = list(size=5)))+
            #change the y-intercept if needed
            ylim(0,ylim_max)+
            theme(panel.grid.major = element_blank(),
                  legend.position = "none",
                  panel.grid.minor = element_blank(),
                  strip.background = element_blank(),
                  panel.border = element_blank(),
                  panel.background = element_rect(fill='white'),
                  plot.subtitle = element_text(hjust = 1),
                  axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
            facet_wrap(.~Sci_SampleID)+
          ggtitle(label = paste0("timepoint",levels(as.factor(ratio_df$Timepoint))[i]),
                  subtitle = paste0("red = user threshold ",user_threshold))
          
          )
  }
  dev.off()
}

mrid_final_plot <- function(input_data, output_path, user_well_outlier_method, user_live_threshold, 
                            user_dead_threshold,ylim_max){
  #make final ratio plots, filter well by mean/median/none 
  
  #save final outputs in another folder
  final_output_path = file.path(output_path,'final_outputs')
  create_output_dir(final_output_path)
  
  # Set width for plot based on number of wells
  n_wells <- length(levels(input_data$Sci_WellID))
  width_manual <- if (n_wells > 96) 25 else 15
  
  # Get filtered data
  date <- Sys.Date()
  mrid_filter_df = mrid_filter(input_data)
  
  # Plot outliers removed by mean
  if(user_well_outlier_method == 'mean'){
    final_data = mrid_filter_df[[3]]
    pdf(file = file.path(final_output_path,paste0("mrid_ratio_outliers_by_mean_final_",date,".pdf")),width = width_manual, height = 8)
    for(i in 1:length(levels(as.factor(final_data$Timepoint)))){
      tmp <- final_data[final_data$Timepoint == levels(as.factor(input_data$Timepoint))[i],]
      
      SD = round(sd(tmp$ratio, na.rm = TRUE),2)
      
      print(ggplot(data=tmp ,aes(x=Sci_WellID,y=ratio))+
              geom_jitter(size=1, alpha=1/5, aes(colour=factor(Sci_WellID)))+
              #change the threshold line
              geom_hline(yintercept = user_live_threshold, colour='red')+
              geom_hline(yintercept = user_dead_threshold, colour='black')+
              #line for mean
              geom_hline(aes(yintercept = mean(ratio), group = Sci_WellID),colour = 'pink',linetype = "solid")+
              guides(colour = guide_legend(override.aes = list(size=5)))+
              #change the y-intercept based on user
              ylim(0, ylim_max)+
              theme(panel.grid.major = element_blank(),
                    legend.position = "none",
                    panel.grid.minor = element_blank(),
                    strip.background = element_blank(),
                    panel.border = element_blank(),
                    panel.background = element_rect(fill='white'),
                    axis.text.x = element_text(angle = 90))+
              ggtitle(paste0("Outliers wells removed by mean, ratio for timepoint ", 
                             levels(as.factor(ratio_df$Timepoint))[i], 
                             " \nMean(solid pink) : ",round(mean(tmp$ratio),2),
                             " \nuser_live_threshold (red): ",user_live_threshold,
                             " \nuser_dead_threshold (black): ",user_dead_threshold))
            
      )
    }
    dev.off()
    
  }else if (user_well_outlier_method == 'median'){
    final_data = mrid_filter_df[[2]]
    pdf(file = file.path(final_output_path,paste0("mrid_ratio_outliers_by_med_final_",date,".pdf")),width = width_manual, height = 8)
    for(i in 1:length(levels(as.factor(final_data$Timepoint)))){
      tmp <- final_data[final_data$Timepoint == levels(as.factor(input_data$Timepoint))[i],]
      
      SD = round(sd(tmp$ratio, na.rm = TRUE),2)
      
      print(ggplot(data=tmp,aes(x=Sci_WellID,y=ratio))+
              geom_jitter(size=1, alpha=1/5, aes(colour=factor(Sci_WellID)))+
              #change the threshold line
              geom_hline(yintercept = user_live_threshold, colour='red')+
              geom_hline(yintercept = user_dead_threshold, colour='black')+
              #line for median
              geom_hline(aes(yintercept = median(ratio), group = Sci_WellID),colour = 'pink',linetype = "solid")+
              guides(colour = guide_legend(override.aes = list(size=5)))+
              #change the y-intercept based on user
              ylim(0, ylim_max)+
              theme(panel.grid.major = element_blank(),
                    legend.position = "none",
                    panel.grid.minor = element_blank(),
                    strip.background = element_blank(),
                    panel.border = element_blank(),
                    panel.background = element_rect(fill='white'),
                    axis.text.x = element_text(angle = 90))+
              ggtitle(paste0("Outliers wells removed by median, ratio for timepoint ", 
                             levels(as.factor(ratio_df$Timepoint))[i], 
                             " \nMedian(solid pink) : ", round(median(tmp$ratio),2),
                             " \nuser_live_threshold (red): ",user_live_threshold,
                             " \nuser_dead_threshold (black): ",user_dead_threshold))
            
      )
    }
    dev.off()
    
  }else{
    final_data = input_data
    print("User did not choose any filtering for wells removal")
  }
  
  return(final_data)
}

mrid_livedead_csv <- function(input_data, output_path, expt_name, user_live_threshold, user_dead_threshold){
  # generates a live/dead ratio from an input data set and saves the result to a ratio_output.csv
  
  final_output_path = file.path(output_path,'final_outputs')
  
  filenames <- paste0(input_data$Sci_PlateID, "_T", input_data$Timepoint, "_", 
                      input_data$Sci_WellID, "_", input_data$ObjectLabelsFound)
  
  livedead_output <- data.frame(filenames, Sci_WellID = input_data$Sci_WellID, 
                                Sci_SampleID = input_data$Sci_SampleID, 
                                Drug = input_data$Drug, Timepoint = input_data$Timepoint, 
                                ratio = input_data$ratio)
  
  livedead_output$live_guesses <- NA
  livedead_output[livedead_output$ratio < user_live_threshold,]$live_guesses <- 1.0
  livedead_output[livedead_output$ratio > user_dead_threshold,]$live_guesses <- 0.0
  
  livedead_output <- livedead_output[!is.na(livedead_output$live_guesses),]
  livedead_output <- within(livedead_output, {classifier.score.live = ifelse(livedead_output$ratio>user_live_threshold,0.0,1.0)})
  livedead_output <- within(livedead_output, {classifier.score.dead = ifelse(livedead_output$ratio>user_live_threshold,1.0,0.0)})
  
  write.csv(livedead_output,file.path(final_output_path,paste0(expt_name,"_ratio_output.csv")))
  return(livedead_output)
}

mrid_livedead_count <- function(input_data){
  
  group_vars <- c("Sci_SampleID", "Sci_WellID", "Timepoint")
  
  if("Drug" %in% colnames(input_data)){
    group_vars <- c(group_vars,"Drug")
  }
  if("Hour" %in% colnames(input_data)){
    group_vars <- c(group_vars, "Hour")
  }
  
  livedead <- input_data %>% 
    dplyr::group_by(!!!rlang::syms(group_vars)) %>% 
    dplyr::summarise(live_guesses = sum(live_guesses), 
                     classifier.score.dead=sum(classifier.score.dead))
  
  livedead$rown <- unlist(lapply(1:length(livedead$Sci_WellID), function(x)
    strsplit(as.character(livedead$Sci_WellID[x]),'[0-30]+')[[1]][1]))
  livedead$coln <- unlist(lapply(1:length(livedead$Sci_WellID), function(x)
    strsplit(as.character(livedead$Sci_WellID[x]),'[a-zA-Z]+')[[1]][2]))

  return(livedead)
}

mrid_heatmap_plot <- function(livedead, output_path, cell_red_flag, perc_change_thresh){
  #make heatmaps that shows the difference and percentage change in live cell counts between timepoints in a plate reader experiment
  # 1)number of live cell per timepoint
  # 2)number of dead cell per timepoint
  #  --> mrid_heatmap_cell_count.pdf
  
  # 3) changes of live cell from previous timepoint
  # 4) percenatge changes of live cell from previous timepoint
  # ---> mrid_heatmap_pctchanges.pdf
  
  livedead_count <- mrid_livedead_count(livedead)
  livedead_count$rown <- factor(livedead_count$rown)
  
  n_timepoints <- length(levels(as.factor(livedead_count$Timepoint)))
  
  # Create PDF file to save the plots
  pdf(file = file.path(output_path,"mrid_heatmap_cell_count.pdf"),width = 10, height = 10)
  
  for (i in 1:n_timepoints){
    tmp <- livedead_count[livedead_count$Timepoint == levels(as.factor(livedead_count$Timepoint))[i],]
    
    p1 <- ggplot(tmp, aes(y = factor(rown, rev(levels(as.factor(rown)))), 
                          x = factor(coln), fill = live_guesses)) + 
      geom_tile() +
      scale_fill_gradient2(low = "#075AFF", mid = "#FFFFCC", high = "#FF0000") +
      labs(x = NULL, y = NULL, fill = 'live cells') +
      geom_text(aes(label = live_guesses), color = "black", size = 4) +
      theme_minimal() +
      coord_fixed() +
      ggtitle(paste0("Number of live cells for timepoint ",levels(as.factor(livedead_count$Timepoint))[i]))
    
    p2 <- ggplot(tmp, aes(y = factor(rown, rev(levels(as.factor(rown)))), 
                          x = factor(coln), fill = classifier.score.dead)) + 
      geom_tile() +
      scale_fill_gradient2() +
      labs(x = NULL, y = NULL, fill = 'dead cells') +
      geom_text(aes(label = classifier.score.dead), color = "black", size = 4) +
      theme_minimal() +
      coord_fixed() +
      ggtitle(paste0("Number of dead cells for timepoint ", levels(as.factor(livedead_count$Timepoint))[i]))
    
    gA <- ggplotGrob(p1)
    gB <- ggplotGrob(p2)
    grid::grid.draw(rbind(gA, gB))
    grid::grid.newpage()
    
  }
  dev.off()
  
  pct_changes <- livedead_count %>%
    group_by(Sci_WellID)  %>%
    dplyr::mutate(lag = lag(live_guesses)) %>%
    dplyr::mutate(diff = live_guesses - lag) %>%
    dplyr::mutate(pct.change = ifelse(lag == 0, diff*100, round((diff/lag) * 100, 1)))
  
  
  # Create the heatmap plot
  
  # Create PDF file to save the plots
  pdf(file = file.path(output_path,"mrid_heatmap_pctchanges.pdf"),width = 10, height = 10)
  for (i in 2:n_timepoints){
    tmp <- pct_changes[pct_changes$Timepoint == levels(as.factor(pct_changes$Timepoint))[i],]
    
    p1 <- ggplot(tmp, aes(y = factor(rown, rev(levels(as.factor(rown)))), 
                          x = factor(coln), fill = diff)) + 
      geom_tile() + theme_minimal() + scale_fill_gradient2(low = "#075AFF", mid = "#FFFFCC", high = "#FF0000") + 
      labs(x = NULL, y = NULL) + 
      geom_text(aes(label = diff), color = "black", size = 4) + 
      theme(text = element_text(size = 17)) + 
      coord_fixed() + 
      ggtitle(paste0("Changes in number of live cells T", levels(as.factor(pct_changes$Timepoint))[i], " and previous tp"))
    
    p2 <- ggplot(tmp, aes(y = factor(rown, rev(levels(as.factor(rown)))), x = factor(coln), fill = diff)) + 
      geom_tile() + theme_minimal() + 
      scale_fill_gradient2(low = "#075AFF", mid = "#FFFFCC", high = "#FF0000") + 
      geom_text(data = tmp, aes(label = pct.change), color = "black", size = 2) + 
      geom_text(data = tmp, aes(label = ifelse(live_guesses < cell_red_flag, as.character(pct.change), "")), color = "red", size = 2) + 
      theme(text = element_text(size = 17)) + 
      coord_fixed() + 
      ggtitle(paste0("% Changes in number of live cells T", levels(as.factor(pct_changes$Timepoint))[i], " and previous tp")) + 
      labs(x = NULL, y = NULL, caption = paste0("Red = well has less than ", cell_red_flag, " cells"))
    
    gA <- ggplotGrob(p1)
    gB <- ggplotGrob(p2)
    grid::grid.draw(rbind(gA, gB))
    grid::grid.newpage()
    
  }
  dev.off()
  
  pct_changes <- pct_changes %>%
    group_by(Sci_WellID) %>%
    mutate(
      large_change = abs(pct.change) > perc_change_thresh,
      consecutive_large_change = lead(large_change, order_by = Timepoint, default = FALSE) &
        large_change &
        lag(large_change, order_by = Timepoint, default = FALSE),
      num_consecutive_large_changes = cumsum(consecutive_large_change)
    ) %>%
    ungroup()  # Remove grouping
  
  
  # Filter rows with consecutive large changes
  filt <-  pct_changes %>%
    group_by(Sci_WellID) %>%
    filter(
      any(consecutive_large_change)
    ) %>%
    ungroup()
  
  filtered_pct_changes <- pct_changes %>%
    anti_join(filt, by = "Sci_WellID")
  
  
  pdf(file = file.path(output_path,"mrid_heatmap_pctchanges_filter.pdf"),width = 15, height = 10)
  for (i in 2:n_timepoints){
    all_sci_well_ids <- unique(pct_changes$Sci_WellID)
    all_rown <- unique(pct_changes$rown)
    all_coln <- unique(pct_changes$coln)
    
    tmp <- pct_changes[pct_changes$Timepoint == levels(as.factor(pct_changes$Timepoint))[i],]
    tmp_filter <- filtered_pct_changes[filtered_pct_changes$Timepoint == levels(as.factor(filtered_pct_changes$Timepoint))[i],]
    
    
    # Set the levels of Sci_WellID in both dataframes
    tmp$Sci_WellID <- factor(tmp$Sci_WellID, levels = all_sci_well_ids)
    tmp_filter$Sci_WellID <- factor(tmp_filter$Sci_WellID, levels = all_sci_well_ids)
    tmp_filter$rown <- factor(tmp_filter$rown, levels = all_rown)
    tmp_filter$coln <- factor(tmp_filter$coln, levels = all_coln)
    
    tmp$bins <- cut(tmp$pct.change, 
                    breaks = c(-Inf, -20, -10, 10, 20, Inf), 
                    labels = c("> -20", "-10 to -20", "-10 to 10", "10 to 20", "< 20"))
    
    
    tmp_filter$bins <- cut(tmp_filter$pct.change, 
                    breaks = c(-Inf, -20, -10, 10, 20, Inf), 
                    labels = c("> -20", "-10 to -20", "-10 to 10", "10 to 20", "< 20"))
    
    my_palette_bins <- c("> -20" = "#FF0000", "-10 to -20" = "pink", "-10 to 10" = "white","10 to 20" = "pink","< 20" = "#FF0000")
    
    # Create the plots with Sci_WellID levels preserved
    p1 <- ggplot(tmp, aes(y = factor(rown, rev(levels(as.factor(rown)))), x = factor(coln), fill = bins)) + 
      geom_tile() + theme_minimal() + 
      scale_fill_manual(values = my_palette_bins) + 
      geom_text(aes(label = pct.change), color = "black", size = 3) +
      labs(x = NULL, y = NULL, fill = "Pct. Change") +
      ggtitle(paste0("% Changes in T", levels(as.factor(pct_changes$Timepoint))[i], " and previous tp"))
    
    complete_data <- expand.grid(rown = all_rown, coln = all_coln)
    
    # Merge the filtered data into the complete dataset
    tmp_filter_complete <- merge(complete_data, tmp_filter, by = c("rown", "coln"), all.x = TRUE)
    
    # Fill any missing values with appropriate defaults
    tmp_filter_complete[is.na(tmp_filter_complete$bins), "bins"] <- NA  # Fill missing bins with "NA"
    tmp_filter_complete[is.na(tmp_filter_complete$pct.change), "pct.change"] <- NA  # Fill missing pct.change with 0
    
  
    my_palette_bins["NA"] <- "transparent"
    
    # Create the ggplot with transparent NA values
    p2 <- ggplot(tmp_filter_complete, aes(y = factor(rown, rev(levels(as.factor(rown)))), x = factor(coln), fill = bins)) +
      geom_tile() + theme_minimal() + 
      scale_fill_manual(values = my_palette_bins, na.value = "transparent") + 
      geom_text(aes(label = pct.change), color = "black", size = 3) +
      labs(x = NULL, y = NULL, fill = "Pct. Change") +
      ggtitle(paste0("% Changes filtered for >= 3 consecutive 15% change in T", levels(as.factor(pct_changes$Timepoint))[i], " and previous tp"))
    
    gA <- ggplotGrob(p1)
    gB <- ggplotGrob(p2)
    grid::grid.draw(rbind(gA, gB))
    grid::grid.newpage()
    
  }
  dev.off()
  
}
