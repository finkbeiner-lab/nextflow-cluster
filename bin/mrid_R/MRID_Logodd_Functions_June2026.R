#' @title MRID RMD functions
#' @author Julia Kaye, Reuben Thomas, Stephanie Lam
#'
#' @description This functions makes MRID log odd stats
#'
#' @import ggplot2
#' @import dplyr
#' @import xtable
#' @import lme4
#'
#' @param input_path contains all csv files
#' @param ratio_file_path mrid ratio csv path
#' @param timepoint_hours  elapsed hour csv path(Column: Timepoint, Hour)
#' @param multi_exp TRUE/FALSE
#' @param multi_plates TRUE/FALSE
#' @param compare_class TRUE/FALSE
#' @param compare_cellline TRUE/FALSE
#' @param compare_drug TRUE/FALSE
#' @param classes list of control and disease string pattern, eg: c("CTR","ALS")
#' @param 
#' 
#' @return 
#'
#' @export
#'
#' @examples logodd_data = mrid_get_logodd(input_path, ratio_file_path, plate_layouts, timepoint_hours, multi_exp ,multi_plates)
#' @examples logodd_data = finalize_df(logodd_data, drug, keep_dose, ref_condition_list, compare_class, classes)
#' @examples mrid_death_plot(logodd_data, drug, multiple_experiments, multiple_plates,  colorpalette)
#' @examples fit_data = mrid_generate_fitdata(logodd_data)
#' @examples mrid_lmer_condition_comparisons(fit_data, ref_condition_list, output_path, compare_cellline, compare_class, compare_drug, multiple_experiments, multiple_plates)
#' @examples mrid_logodd_combine_plot(odd_ratio_data, reference_condition, output_path, colorpalette, confident_int = 0.95)


#convert old CNN_output.csv to new ratio.csv
convert_old_cnnoutput <- function(input_path){
  cnn_files <- list.files(path = input_path, pattern = "CNN_output[0-9_]*?.csv", recursive = TRUE)
  
  if (length(cnn_files) != 0) {
    plate_layout_files <- list.files(path = input_path, pattern = "[plate]*?layout.csv", recursive = TRUE)
    
    for(i in 1:length(cnn_files)){
      ratio_csv <- read.csv(file.path(input_path, cnn_files[i]))
      subfolder_check = if(grepl("/", cnn_files[i])){subfolder = strsplit(cnn_files[i],'/')[[1]][[1]]}else{subfolder_check=''}
        
      file_string <- strsplit(basename(cnn_files[i]),"_CNN_output.csv")[[1]][1]
      platelayout <- read.csv(file.path(input_path,plate_layout_files[which(grepl(file_string,plate_layout_files))]), header = TRUE)
      
      ratio_csv$Sci_WellID <- as.character(lapply(ratio_csv$filenames, function(x) strsplit(x,'_')[[1]][4]))
      ratio_csv$Timepoint <- as.numeric(gsub("T","",lapply(ratio_csv$filenames, function(x) strsplit(x,'_')[[1]][3])))
      ratio_csv <- as.data.frame(ratio_csv)
      platelayout <- as.data.frame(platelayout)
      
      ratio_csv <- merge(ratio_csv,platelayout)
      
      ratio_csv <- ratio_csv %>% select("filenames","Sci_WellID","Sci_SampleID","Drug",
                                        "Timepoint","ratio","live_guesses","classifier.score.live","classifier.score.dead")
      write.csv(ratio_csv,paste0(input_path,'/',subfolder_check,'/',file_string,"_ratio_output.csv"))
      
      print(paste0("Converting old", cnn_files[i]," to ",file_string,"_ratio_output.csv"))
      
    }
  }
}


#get logodd data
mrid_get_logodd <- function(input_path, ratio_file_path, timepoint_hours, multi_exp, multi_plates){
  
  mrid_read_data <- function(input_path, ratio_file_path, timepoint_hours){
    
    ratio_csv <- read.csv(file.path(input_path, ratio_file_path))
    
    file_string <- str_remove(basename(ratio_file_path), "_ratio_output.csv")
    elapsed_hours <- read.csv(file.path(input_path,timepoint_hours[which(grepl(file_string,timepoint_hours))]), header = TRUE)
    
    names(elapsed_hours)[1] <- "Timepoint"
    elapsed_hours$Timepoint <- as.numeric(gsub("T","",elapsed_hours$Timepoint))
    names(elapsed_hours)[2] <- "Hour"
    elapsed_hours$Hour <- as.numeric(elapsed_hours$Hour)
    
    ratio_csv <- merge(ratio_csv,elapsed_hours)
    
    return(ratio_csv)
  }
  

  ratio_csv <- mrid_read_data(input_path, ratio_file_path, timepoint_hours)
  
  livedead_count = mrid_livedead_count(ratio_csv)
  
  #calculate death
  FullData <- data.frame(livedead_count, 
                         Death = livedead_count$classifier.score.dead/
                           (livedead_count$live_guesses + livedead_count$classifier.score.dead))
  
  # # setting death @T0 to 0
  # FullData$Death <- sapply(seq_len(nrow(FullData)), function(x) {
  #   if (FullData$Timepoint[x] == min(FullData$Timepoint)) { 0.0 } else { FullData$Death[x] }
  # })
  
  #calculate log odd death
  FullData <- data.frame(FullData, logOddsDeath = log(FullData$Death / (1 - FullData$Death)))
  FullData <- FullData[!is.na(FullData$live_guesses),]
  
  if (multiple_experiments) {
    #experiment name from folder name
    experiment_name <-  ifelse(multiple_plates, dirname(dirname(ratio_file_path)), dirname(ratio_file_path))
    plate_name <- ifelse(multiple_plates, basename(dirname(ratio_file_path)), NA)
    
  } else {
    experiment_name <- ifelse(multiple_plates, basename(input_path), strsplit(basename(ratio_file_path),"_ratio_output.csv")[[1]][1])
    plate_name <- ifelse(multiple_plates, dirname(ratio_file_path), NA)
  }
    
  FullData$Experiment <- experiment_name
  FullData$Plate <- plate_name
  
  FullData <- FullData %>% 
    mutate(Experiment = experiment_name, Plate = plate_name)
  
  return(FullData)
}


#finalize logodd dataframe, and filter dose if needed
finalize_df <- function(data, drug, keep_dose, keep_line, keep_line_drug, ref_condition_list, compare_class, multiple_experiments, classes){
  #data= logodd_data
  
  #Filter out wells with zero live cells and wells without Sci_SampleID
  data <- data[data$live_guesses != 0, ]
  data <- data[!is.na(data$Sci_SampleID), ]
  
  #handle mismatch hours
  unique_hours <- sort(unique(factor(data$Hour)))
  hour_mapping <- setNames(1:length(unique_hours), unique_hours)
  data$Timepoint <- hour_mapping[as.factor(data$Hour)]
  
  #handle and rearrange timepoints
  timepoint_unique <- unique(data$Timepoint)
  data$Timepoint <- factor(data$Timepoint, levels = sort(as.integer(timepoint_unique)))
  
  #combine condition and drug
  data$Drug <- gsub("[/_]"," ",data$Drug)
  data$CellLine <- if(drug){paste0(data$Sci_SampleID, " ", data$Drug)}else{data$Sci_SampleID}
  
  ref_condition_list <- gsub("[/_]"," ",ref_condition_list)
  #matches <- paste(ref_condition_list, collapse = "|")
  
  if (!(length(keep_drug) == 0 || all(keep_drug == ""))) {
    # Filter based on the values in keep_drug
    data <- data[data$Drug %in% keep_drug, ]
  }
  if (!(length(keep_line) == 0 || all(keep_line == ""))) {
    # Filter based on the values in keep_line
    data <- data[data$Sci_SampleID %in% keep_line, ]
  }
  if (!(length(keep_line_drug) == 0 || all(keep_line_drug == ""))) {
    if (length(keep_line_drug) < 2) {
      stop("keep_line_drug must have more than one CellLine.")
    }else{# Filter based on the values in keep_line
      data <- data[data$CellLine %in% keep_line_drug, ]
    }
  }
  
  #data <- data[grepl(matches, data$CellLine),]
  
  if(compare_class){
    if (all(!is.na(classes))){
      for(i in 1:length(classes)){
        data$Class[grepl(classes[i], data$Sci_SampleID, ignore.case=FALSE)] <- i-1
      }
      data$Class <- factor(data$Class)
    }else{
      stop("Please label classes in user input")
    }
  }
  
  if(multiple_experiments){
      single_exp = data %>% group_by(Sci_SampleID) %>% summarise(count = n_distinct(Experiment)) %>% filter(count == 1)
      if(dim(single_exp)[1]>0){
        cat("Removing Sci_SampleIDs that only appears in one experiment\n")
        print(knitr::kable(single_exp))
        data = data %>% filter(!(Sci_SampleID %in% single_exp$Sci_SampleID))
        data$Sci_SampleID = factor(data$Sci_SampleID)
        cat("\n")
      }
      single_tp = data %>% group_by(Hour) %>% summarise(count = n_distinct(Experiment)) %>% filter(count == 1)
      if(dim(single_tp)[1]>0){
        cat("Removing Hours that only appears in one experiment\n")
        print(knitr::kable(single_tp))
        data = data %>% filter(!(Hour %in% single_tp$Hour))
        data$Hour = factor(data$Hour)
        cat("\n")
      }
  }
  
  #factorize columns
  columns_to_factorize <- c("Sci_SampleID", "Experiment", "Plate", "CellLine", "Hour")
  data[, columns_to_factorize] <- lapply(data[, columns_to_factorize], factor)

  print(knitr::kable(xtable(data %>% group_by(Experiment, Plate, CellLine) %>% 
                 dplyr::summarise(sum_live = sum(live_guesses), 
                                  sum_dead = sum(classifier.score.dead)), 
               caption = "Total Live/dead per condition",
               format = "markdown")))
  
  print(knitr::kable(xtable(data %>% group_by(Experiment, Plate, CellLine , Timepoint, Hour) %>% 
                              dplyr::summarise(sum_live = sum(live_guesses), 
                                               sum_dead = sum(classifier.score.dead)), 
                            caption = "Total Live/dead per condition per timepoint",
                            format = "markdown")))
  
  if(compare_class){
    print(knitr::kable(xtable(data %>% group_by(Experiment, Plate, Class, Timepoint) %>% 
                                dplyr::summarise(sum_live = sum(live_guesses), 
                                                 sum_dead = sum(classifier.score.dead)), 
                              caption = "Total Live/dead per condition",
                              format = "markdown")))
  }
  
  print(knitr::kable(xtable(data  %>% group_by(Experiment) %>%
                              dplyr::summarise(sum_live = sum(live_guesses), 
                                               sum_dead = sum(classifier.score.dead)), 
                            caption = "Total Live/dead Per Experiment",
                            format = "markdown")))
  return(data)
}

#function to plot death rate
mrid_death_plot <- function(data, drug, multiple_experiments, multiple_plates, colorpalette, compare_class = FALSE){
  #data = logodd_data
  base_plot <- ggplot(data, aes(Hour, Death, color = CellLine)) +
    geom_boxplot() +
    ylim(0, 1) +
    theme(legend.position = "top",
          legend.key.size = unit(0.2, "cm"))+
    #theme(text = element_text(size = 14)) +
    ggtitle("Death rate")
  
  if(!is.null(colorpalette)){
    if(length(names(colorpalette)) != length(unique(logodd_data$CellLine))){
      missing_colors = unique(logodd_data$CellLine)[!unique(logodd_data$CellLine) %in% names(colorpalette)]
      print(paste0("Color lengths not matching with length of conditions, Please add color manually for: ",
                   missing_colors))
      for (color in missing_colors) {
        if (!(color %in% names(colorpalette))) {
          colorpalette[[color]] <- sample(colors(), 1)
          print(paste0("Automatically generated color for: ", color))
        }
      }
    }
    names(colorpalette) <- gsub("[/_]"," ",names(colorpalette))
    base_plot <- base_plot + scale_color_manual(values = colorpalette)
  }
  
  
  if (multiple_plates) {
    print(base_plot +
            facet_wrap(Plate ~ Experiment, scales = "free"))
  } else {
    print(base_plot +
            facet_wrap(~Experiment, scales = "free"))
  }
  
  overall_death_plot <- base_plot + 
    aes(if(drug) {Drug} else {Sci_SampleID}, Death, color = CellLine) +
    ggtitle(paste0("Overall death rate per ", if(drug) {"drug"} else {"Sci_SampleID"})) +
    xlab(if(drug) {"Drug"} else {"Sci_SampleID"})
  print(overall_death_plot)
  
  if(multiple_experiments){
    n_exp = length(levels(factor(data$Experiment)))
      for(i in 1:range(n_exp)){
        base <- ggplot(data[data$Experiment == levels(data$Experiment)[i],], aes(Hour, Death, color = CellLine)) +
          geom_boxplot() +
          ylim(0, 1) +
          theme(legend.position = "top") +
          theme(text = element_text(size = 17)) +
          ggtitle(paste0("Death rate per experiment: ",levels(data$Experiment)[i]))
        
        if(!is.null(colorpalette)){
          base <- base + scale_color_manual(values = colorpalette)
        }
        
        if(multiple_plates){
          print( base + facet_wrap(~Plate,scales = "free"))
        }else{
         print(base)
      }
      }
    print(base_plot + ggtitle("Combined Experiments"))
    
  }
  if(compare_class){
    class_color_palette <- c('0' = 'grey', '1' = 'red')
    
    #assign colors if more one 2 classes
    unique_values <- unique(data$Class)
    if (!all(unique_values %in% names(class_color_palette))) {
      other_values <- setdiff(unique_values, names(class_color_palette))
      num_other_values <- length(other_values)
      
      # Create a gradient of red colors
      other_colors <- scales::col_numeric(
        palette = c("pink", "darkred"), 
        domain = range(unique_values)
      )(other_values)
      
      # Merge the custom color palette
      class_color_palette <- c(class_color_palette, setNames(other_colors, as.character(other_values)))
    }
  
    print(ggplot(data, aes(Hour, Death, color = Class)) + geom_boxplot() +
            ylim(0, 1) +
            theme(legend.position = "top") +
            theme(text = element_text(size = 17)) +
            ggtitle("Overall death rate (Class)")+ scale_color_manual(values = class_color_palette))
    
    if(length(levels(factor(data$Drug)))>1){
      print(ggplot(data, aes(Hour, Death, color = Class)) + geom_boxplot() + facet_grid(.~Drug)+
              ylim(0, 1) +
              theme(legend.position = "top") +
              theme(text = element_text(size = 17)) +
              ggtitle("Overall death rate (Class + Drug)")+ scale_color_manual(values = class_color_palette))
    }
  }
  
  print(ggplot(data, aes(Hour, Death, color = Sci_SampleID)) + geom_boxplot() +
      ylim(0, 1) +
        theme(legend.position = "top") +
        theme(text = element_text(size = 17)) +
      ggtitle("Overall death rate (Cell line)"))
  
  if(length(levels(factor(data$Drug)))>1){
    print(ggplot(data, aes(Hour, Death, color = Drug)) + geom_boxplot() + facet_grid(.~Sci_SampleID)+
          ylim(0, 1) + theme(legend.position = "top") +
            theme(text = element_text(size = 17)) +
          ggtitle("Overall death rate (Cell line + Drug)"))
  }
  if(length(levels(factor(data$Drug)))>1 & length(levels(factor(data$Sci_SampleID))) > 3){
    for(i in 1:length(levels(factor(data$Sci_SampleID)))){
      tmp <- data[data$Sci_SampleID == levels(factor(data$Sci_SampleID))[i],]
      print(ggplot(tmp, aes(Hour, Death, color = Drug)) + geom_boxplot() + facet_grid(.~Sci_SampleID)+
              ylim(0, 1) +
              theme(legend.position = "top", legend.key.size = unit(0.2, "cm")) + 
              ggtitle("Overall Death Rate Drug Per Cell Lines"))
    }
  }
  
}

#function to plot residuals
mrid_residuals_plot <- function(data, drug, multiple_experiments, multiple_plates, colorpalette, compare_class = FALSE, compare_cellline, compare_drug, TimepointIsNumeric){
  #data = logodd_data
  #glmerfit0 <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Sci_SampleID), logodd_data, family = "binomial")
  
  glmerfit0 <- mrid_lmer_residual(data,compare_cellline, compare_class, compare_drug, multiple_experiments, multiple_plates, TimepointIsNumeric)
  resid_odds <- residuals(glmerfit0)
  data$Timepoint <- as.factor(data$Timepoint)
  
  plot_data <- data.frame(logodd_data, resid_odds)
  
  # ggplot(plot_data, aes(x = Timepoint, y=resid_odds, color = ID)) + 
  #   geom_boxplot() +
  #   theme(legend.position = "top") +
  #   theme(text = element_text(size = 17)) +
  #   theme_bw()+
  #   ggtitle("Residuals plot per cell lines")
  
  if(length(levels(factor(data$Drug)))>1){
    ggplot(plot_data, aes(x = Timepoint, y=resid_odds, color = Drug)) + 
      geom_boxplot() +
      theme(legend.position = "top") +
      theme(text = element_text(size = 17)) +
      theme_bw()+
      ggtitle("Residuals plot per drug")
  }
  
  if(multiple_experiments){
    ggplot(plot_data, aes(x = Timepoint, y=resid_odds, color = Class)) + 
      geom_boxplot() +
      theme(legend.position = "top") +
      theme(text = element_text(size = 17)) +
      theme_bw()+
      ggtitle("Residuals plot per experiment")+
      facet_wrap(~Experiment)
  }
  
  
  if(compare_class){
    
    cat('------------------------------------------------------------------------ ')
    
    base_plot <- ggplot(plot_data, aes(x = Timepoint, y=resid_odds, color = Class)) + 
      geom_boxplot() +
      theme(legend.position = "top") +
      theme(text = element_text(size = 17)) +
      theme_bw()+
      ggtitle("Residuals plot")
    print(base_plot)
  
    class_color_palette <- c('0' = 'grey', '1' = 'red')
    
    #assign colors if more one 2 classes
    unique_values <- unique(data$Class)
    if (!all(unique_values %in% names(class_color_palette))) {
      other_values <- setdiff(unique_values, names(class_color_palette))
      num_other_values <- length(other_values)
      
      # Create a gradient of red colors
      other_colors <- scales::col_numeric(
        palette = c("pink", "darkred"), 
        domain = range(unique_values)
      )(other_values)
      
      # Merge the custom color palette
      class_color_palette <- c(class_color_palette, setNames(other_colors, as.character(other_values)))
    }
    
    print(ggplot(plot_data, aes(Hour, y=resid_odds, color = Class)) + geom_boxplot()+
            theme(legend.position = "top") +
            theme(text = element_text(size = 17)) +
            ggtitle("Residuals plot")+ scale_color_manual(values = class_color_palette))
    
    if(length(levels(factor(data$Drug)))>1){
      print(ggplot(plot_data, aes(Hour, y=resid_odds, color = Class)) + geom_boxplot() + facet_grid(.~Drug)+
              theme(legend.position = "top") +
              theme(text = element_text(size = 17)) +
              ggtitle("Residuals plot (Class + Drug)")+ scale_color_manual(values = class_color_palette))
    }
  }
}


#functions for lmer model ---------

#lmer models
mrid_lmer <- function(data, compare_cellline, compare_class, compare_drug, multiple_experiments, multiple_plates, TimepointIsNumeric = TRUE){
  #data = fit_data
  data$CellLine <- factor(data$CellLine)
  if(TimepointIsNumeric)
    data$Timepoint <- as.numeric(data$Timepoint)
  else
    data$Timepoint <- as.factor(data$Timepoint)
  
  if (compare_class){
    data$Class <- factor(data$Class)
    if(multiple_experiments){
      if(multiple_plates){
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Experiment:Plate) + (1|CellLine) + Class + Timepoint + Timepoint:Class, data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + (1|CellLine) + Class + Timepoint + Timepoint:Class, data, family = "binomial")
      }
    }else{
      if(multiple_plates){
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Plate) + (1|Plate:Sci_WellID) + (1|CellLine) + Class + Timepoint + Timepoint:Class, data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|CellLine) +(1|Sci_WellID) + Class + Timepoint + Timepoint:Class, data, family = "binomial")
      }
    }
  }else if(compare_cellline && compare_drug){
    data$Drug <- as.numeric(data$Drug)
    if(multiple_experiments){
      if(multiple_plates){
        glmFit <-glmer(cbind(classifier.score.dead, live_guesses) ~  (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Experiment:Plate) + (1|CellLine)  + Drug + Timepoint + Drug:Timepoint, data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + (1|CellLine)  + Drug + Timepoint + Drug:Timepoint, data, family = "binomial")
      }
    }else{
      if(multiple_plates){
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Plate) + (1|CellLine)  + Drug + Timepoint + Drug:Timepoint, data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|CellLine)  + Drug + Timepoint + Drug:Timepoint, data, family = "binomial")
      }
    }
  }else if(compare_cellline){
    #single line comparison
    if(multiple_experiments){
      if(multiple_plates){
        glmFit <-glmer(cbind(classifier.score.dead, live_guesses) ~  (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Experiment:Plate) + CellLine + Timepoint + Timepoint:CellLine, data, family = "binomial")
      }else{
        glmFit <-glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + CellLine + Timepoint + Timepoint:CellLine, data, family = "binomial")
      }
    }else{
      if(multiple_plates){
        glmFit <-glm(cbind(classifier.score.dead, live_guesses) ~ Plate + CellLine + Timepoint + Timepoint:CellLine, data, family = "binomial")
      }else{
        glmFit <-glm(cbind(classifier.score.dead, live_guesses) ~ CellLine + Timepoint + Timepoint:CellLine, data, family = "binomial")
      }
    }
  }
  else if(compare_drug & length(unique(data$CellLine)) == 1){
    #single line comparison
    if(multiple_experiments){
      if(multiple_plates){
        glmFit <-glmer(cbind(classifier.score.dead, live_guesses) ~  (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Experiment:Plate) + Drug + Timepoint + Timepoint:Drug, data, family = "binomial")
      }else{
        glmFit <-glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + Drug + Timepoint + Timepoint:Drug, data, family = "binomial")
      }
    }else{
      if(multiple_plates){
        glmFit <-glm(cbind(classifier.score.dead, live_guesses) ~ Plate + Drug + Timepoint + Timepoint:Drug, data, family = "binomial")
      }else{
        glmFit <-glm(cbind(classifier.score.dead, live_guesses) ~ Drug + Timepoint + Timepoint:Drug, data, family = "binomial")
      }
    }
  }
  
  return(glmFit)
}

mrid_lmer0 <- function(data, compare_cellline, compare_class, compare_drug, multiple_experiments, multiple_plates, TimepointIsNumeric = TRUE){
  #data = fit_data
  data$CellLine <- factor(data$CellLine)
  if(TimepointIsNumeric)
    data$Timepoint <- as.numeric(data$Timepoint)
  else
    data$Timepoint <- as.factor(data$Timepoint)
  
  if (compare_class){
    data$Class <- factor(data$Class)
    if(multiple_experiments){
      if(multiple_plates){
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Experiment:Plate) + (1|CellLine) +  Timepoint , data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + (1|CellLine)  + Timepoint , data, family = "binomial")
      }
    }else{
      if(multiple_plates){
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Plate) + (1|Plate:Sci_WellID) + (1|CellLine)  + Timepoint , data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|CellLine) +(1|Sci_WellID)  + Timepoint , data, family = "binomial")
      }
    }
  }else if(compare_cellline && compare_drug){
    data$Drug <- as.numeric(data$Drug)
    if(multiple_experiments){
      if(multiple_plates){
        glmFit <-glmer(cbind(classifier.score.dead, live_guesses) ~  (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Experiment:Plate) + (1|CellLine)   + Timepoint , data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + (1|CellLine)   + Timepoint , data, family = "binomial")
      }
    }else{
      if(multiple_plates){
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Plate) + (1|CellLine)   + Timepoint , data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|CellLine)   + Timepoint,  data, family = "binomial")
      }
    }
  }else if(compare_cellline | compare_drug){
    #single line comparison
    if(multiple_experiments){
      if(multiple_plates){
        glmFit <-glmer(cbind(classifier.score.dead, live_guesses) ~  (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Experiment:Plate)  + Timepoint , data, family = "binomial")
      }else{
        glmFit <-glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID)  + Timepoint , data, family = "binomial")
      }
    }else{
      if(multiple_plates){
        glmFit <-glm(cbind(classifier.score.dead, live_guesses) ~ Plate  + Timepoint , data, family = "binomial")
      }else{
        glmFit <-glm(cbind(classifier.score.dead, live_guesses) ~   Timepoint , data, family = "binomial")
      }
    }
  }
  
  return(glmFit)
}

mrid_lmer_residual <- function(data, compare_cellline, compare_class, compare_drug, multiple_experiments, multiple_plates, TimepointIsNumeric = TRUE){
  #data = fit_data
  data$CellLine <- factor(data$CellLine)
  if(TimepointIsNumeric)
    data$Timepoint <- as.numeric(data$Timepoint)
  else
    data$Timepoint <- as.factor(data$Timepoint)
  
  if (compare_class){
    data$Class <- factor(data$Class)
    if(multiple_experiments){
      if(multiple_plates){
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Experiment:Plate) + (1|CellLine)  , data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + (1|CellLine)  , data, family = "binomial")
      }
    }else{
      if(multiple_plates){
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Plate) + (1|Plate:Sci_WellID) + (1|CellLine)  , data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|CellLine) +(1|Sci_WellID)  , data, family = "binomial")
      }
    }
  }else if(compare_cellline && compare_drug){
    data$Drug <- as.numeric(data$Drug)
    if(multiple_experiments){
      if(multiple_plates){
        glmFit <-glmer(cbind(classifier.score.dead, live_guesses) ~  (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Experiment:Plate) + (1|CellLine)  , data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID) + (1|CellLine) , data, family = "binomial")
      }
    }else{
      if(multiple_plates){
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Plate) + (1|CellLine) , data, family = "binomial")
      }else{
        glmFit <- glmer(cbind(classifier.score.dead, live_guesses) ~ (1|CellLine) ,  data, family = "binomial")
      }
    }
  }else if(compare_cellline | compare_drug){
    #single line comparison
    if(multiple_experiments){
      if(multiple_plates){
        glmFit <-glmer(cbind(classifier.score.dead, live_guesses) ~  (1|Experiment) + (1|Experiment:Sci_WellID) + (1|Experiment:Plate)  , data, family = "binomial")
      }else{
        glmFit <-glmer(cbind(classifier.score.dead, live_guesses) ~ (1|Experiment) + (1|Experiment:Sci_WellID)  , data, family = "binomial")
      }
    }else{
      if(multiple_plates){
        glmFit <-glm(cbind(classifier.score.dead, live_guesses) ~ Plate , data, family = "binomial")
      }else{
        glmFit <-glm(cbind(classifier.score.dead, live_guesses) ~   1 , data, family = "binomial")
      }
    }
  }
  
  return(glmFit)
}

#get lmer summary stats 

mrid_lmer_stats <- function(data, reference_condition, comparing_condition, output_path, 
                                            compare_cellline, compare_class, compare_drug,
                                            multiple_experiments, multiple_plates, TimepointIsNumeric = TRUE){
  #data = logodd_data
  
  # Filter the data
  if(compare_cellline && compare_drug){
    #input data focus on single drug already
    data$Drug <- relevel(factor(data$Drug), ref = reference_condition)
  }else if (compare_class && compare_drug | compare_class){
    data$Class <- relevel(factor(data$Class), ref = reference_condition)
  }else if(compare_cellline | compare_drug){
    tmp <- data[(data$CellLine == reference_condition | data$CellLine == comparing_condition),]
    if(dim(tmp)[1] == 0){
      data <- data[(data$Drug == reference_condition | data$Drug == comparing_condition),]
      data$Drug <- relevel(factor(data$Drug), ref = reference_condition)
    }else{
      data = tmp
      data$CellLine <- relevel(factor(data$CellLine), ref = reference_condition)
    }
  }
  data$Timepoint <- as.numeric(data$Timepoint)
  
  glmFit <- mrid_lmer(data, compare_cellline, compare_class, compare_drug, multiple_experiments, multiple_plates, TimepointIsNumeric)
  #Summarize the model fit
  sglmFit <- summary(glmFit)
  
  glmFit0 <- mrid_lmer0(data, compare_cellline, compare_class, compare_drug, multiple_experiments, multiple_plates, TimepointIsNumeric)
  #Summarize the model fit
  sglmFit0 <- summary(glmFit0)
  
  anova_res <- anova(glmFit, glmFit0)
  #print(anova_res)
  pval_index <- length(anova_res[2,])
  cat('\n')
  print(paste0("Overall significance (p-value) of association across all time-points = ", anova_res[2, pval_index]))
  cat('\n')
  print(paste0("Akaike Information Criterion (AIC) = ", AIC(glmFit)))
  cat('\n')
  
  tryCatch({
    print(knitr::kable(xtable(sglmFit, caption = paste("glm -", reference_condition, "vs", comparing_condition))))
  }, error = function(e) {
    stats <- as.data.frame(coef(summary(glmFit)))
    print(knitr::kable(stats, caption = paste("glm -", reference_condition, "vs", comparing_condition)))
  })
  
  # Get the coefficients
  Coef <- NULL
  
  #if (multiple_experiments) NExp <- length(levels(data$Experiment))
  if (grepl("Plate +",formula(glmFit)[3])) {
    #check if plate is a fixed effect
    n <- length(levels(factor(data$Plate)))
  }else{
    n = 1
  }
  
  if(!TimepointIsNumeric) {
    NTime <- length(levels(factor(data$Timepoint)))
    if (compare_class) {
      NClass <- length(levels(factor(data$Class)))
      Coef <- sglmFit$coefficients[c((n+1):(n+NClass-1), (n+NClass+NTime-1):(NClass*NTime)),]
    } else if (compare_cellline && compare_drug){
      NDrug <- length(levels(data$Drug))
      Coef <- sglmFit$coefficients[c((n+1):(n+NDrug-1), (n+NDrug+NTime-1):(NDrug*NTime)),]
      NClass <- NDrug
    } else if (compare_cellline | compare_drug) {
      NCellLines <- length(levels(data$CellLine))
      Coef <- sglmFit$coefficients[c((n+1):(n+NCellLines-1), (n+NCellLines+NTime-1):(NCellLines*NTime)),]
      NClass <- NCellLines
    }
    Coef <- as.data.frame(Coef)
    
    ##plot odds ratios over time
    PlotTime <- 0:(length(levels(factor(data$Timepoint))) - 1)
    
    Estimate <- vector(mode = "numeric")
    SE <- vector(mode = "numeric")
    Estimate[1] <- Coef$Estimate[1]
    SE[1] <- sqrt(Coef$`Std. Error`[1]^2)
    VarCovMatrix <- vcov(glmFit)
    ##remove rows and columns of this matrix that may come when Plate is modeled as a fixed effect
    if(sum(grepl("Plate", row.names(VarCovMatrix))) > 0) {
      plate_indices <- grep("Plate", row.names(VarCovMatrix))
      VarCovMatrix <- VarCovMatrix[-plate_indices,]
      VarCovMatrix <- VarCovMatrix[,-plate_indices]
    }
    
    for(t in 2:NTime) {
      Estimate[t] <- Coef$Estimate[1]  + Coef$Estimate[t]
      choose_coefficients <- rep(0, NClass*NTime)
      choose_coefficients[2] <- 1
      choose_coefficients[NTime + NClass + t - 2] <- 1
      SE[t] <- sqrt(t(choose_coefficients)%*%VarCovMatrix%*%(choose_coefficients))
    }
  }
  
  if(TimepointIsNumeric) {
    if (compare_class) {
      NClass <- length(levels(factor(data$Class)))
      Coef <- sglmFit$coefficients[(n + 1):(n + 2 * NClass - 1),]
    } else if (compare_cellline && compare_drug){
      NDrug <- length(levels(data$Drug))
      Coef <- sglmFit$coefficients[(n + 1):(n + 2 * NDrug - 1),]
      NClass <- NDrug
    } else if (compare_cellline | compare_drug) {
      NCellLines <- length(levels(data$CellLine))
      Coef <- sglmFit$coefficients[(n + 1):(n + 2 * NCellLines - 1),]
      NClass <- NCellLines
    }
    Coef <- as.data.frame(Coef)
    
    ##plot odds ratios over time
    PlotTime <- 0:(length(levels(factor(data$Timepoint)))-1)
    
    VarCovMatrix <- vcov(glmFit)
    ##remove rows and columns of this matrix that may come when Plate is modeled as a fixed effect
    if(sum(grepl("Plate", row.names(VarCovMatrix))) > 0) {
      plate_indices <- grep("Plate", row.names(VarCovMatrix))
      VarCovMatrix <- VarCovMatrix[-plate_indices,]
      VarCovMatrix <- VarCovMatrix[,-plate_indices]
    }
    choose_coefficients <- rep(0, NClass*2)
    choose_coefficients[2] <- 1
    
    
    Estimate <- vector(mode = "numeric")
    SE <- vector(mode = "numeric")
    
    Estimate[1] <- Coef$Estimate[1]
    SE[1] <- sqrt(Coef$`Std. Error`[1]^2 )
    for(t in 2:length(PlotTime)) {
      Estimate[t] <- Coef$Estimate[1]  + PlotTime[t] * Coef$Estimate[3]
      choose_coefficients[2*NClass] <- PlotTime[t]
      SE[t] <- sqrt(t(choose_coefficients)%*%VarCovMatrix%*%(choose_coefficients))
    }
  }
  
  PointEstimate <- exp(Estimate)
  Estimate_lower <- exp(Estimate - 1.96 * SE)
  Estimate_upper <- exp(Estimate + 1.96 * SE)
  lower_99 <- exp(Estimate - 2.575 * SE)
  upper_99 <- exp(Estimate + 2.575 * SE)
  pvalue_norm_approx <- 2*pnorm(-abs(Estimate), sd = SE)
  if(compare_cellline | compare_drug)  LCellLines <- levels(factor(data$CellLine))
  if(compare_cellline && compare_drug)  LCellLines <- levels(factor(data$Drug))
  if(compare_class) { LCellLines <- levels(factor(data$Class))}
  
  PlotData <- data.frame(OR=PointEstimate, 
                         Time=PlotTime, 
                         lower=Estimate_lower, 
                         upper=Estimate_upper, 
                         lower99=lower_99, 
                         upper99 = upper_99,
                         pvalue = pvalue_norm_approx,
                         Reference = LCellLines[1],
                         Compare = LCellLines[2])
  
  if(compare_class && compare_drug){
    PlotData['Drug'] <- levels(factor(data$Drug))
    write.csv(PlotData, file.path(output_path,paste0("OR_", LCellLines[1], "_", LCellLines[2],"_",levels(factor(data$Drug)),".csv")))
  }else{
    write.csv(PlotData, file.path(output_path,paste0("OR_", LCellLines[1], "_", LCellLines[2],".csv")))
  }
  
  Plotmean1half <- colMeans(PlotData[1:ceiling(nrow(PlotData) / 2),  1:6]) 
  Plotmean2half <- colMeans(PlotData[(ceiling(nrow(PlotData) / 2) + 1):nrow(PlotData),  1:6]) 
  PlotMean <- data.frame(t(apply(PlotData[1:6], 2, mean)))
  
  print(knitr::kable(PlotData))
  print(knitr::kable(PlotMean))
  
  cat('\n\n')
  cat('First half Mean \n')
  print(knitr::kable(data.frame(t(Plotmean1half))))
  cat('Second half Mean \n')
  print(knitr::kable(data.frame(t(Plotmean2half))))
  
  print(ggplot(PlotData, aes(y=OR, x=Time)) + 
          geom_ribbon(aes(ymin = lower, ymax = upper), fill = "grey70") + 
          geom_hline(yintercept = 1, linetype=2) + 
          geom_line(stat="identity")+ xlab("Time (in days)") +
          theme(text = element_text(size = 17))+
          ggtitle(paste0("Odds ratio of ", LCellLines[2], " vs ", LCellLines[1])))
  cat("\n\n\\pagebreak\n")
}

#main function - run mrid_lmer_stats
mrid_lmer_condition_comparisons <- function(data, ref_condition_list, output_path, 
                                            compare_cellline, compare_class, compare_drug, 
                                            multiple_experiments, multiple_plates, classes_string, TimepointIsNumeric = TRUE) {
  #data = logodd_data
  #cell line - cell line
  if(compare_cellline && !compare_drug) {
    #compare across all different cell line
    #compare_cellline <- TRUE, compare_class <- FALSE, compare_drug <- FALSE
    if(length(ref_condition_list) == 0){
      print("No reference conditions. Comparing all cell lines...")
      ref_condition_list = levels(factor(data$CellLine))
    }
    ref_condition_list <- gsub("[/_]", " ", ref_condition_list)
    for (i in 1:length(ref_condition_list)) {
        comparing_conditions = NULL
  
        if (!ref_condition_list[i] %in% data$CellLine) {
          stop("ERROR: Condition not found! Make sure reference condition is one of the cell line!")
        }
        reference_condition <- ref_condition_list[i]
        comparing_conditions <- setdiff(levels(factor(data$CellLine)), reference_condition)
        for(j in 1:length(comparing_conditions)){
          comparing_condition = comparing_conditions[j]
          
          matching_data <- subset(data, CellLine %in% c(reference_condition, comparing_condition))
          #iterate over conditions and call the mrid_lmer_stats function for each comparison
          
          cat(paste0("\n\n## Odds Ratio Comparing: ", reference_condition, " vs ", comparing_condition, "\n\n"))
          mrid_lmer_stats(matching_data, reference_condition, comparing_condition, output_path, 
                          compare_cellline, compare_class, compare_drug, 
                          multiple_experiments, multiple_plates, TimepointIsNumeric)
        }
    }

  }
  #same drug - different class
  else if(compare_class && compare_drug){
    #compare class within the same drug
    #compare_cellline <- FALSE, compare_class <- TRUE, compare_drug <- TRUE
    conditions <- levels(factor(fit_data$Drug))
    celllines <- levels(factor(fit_data$CellLine))
    classes <- levels(factor(fit_data$Class))
    sub_lists <- list()
    #loop thru classes within same drug 
    for (condition in conditions) {
        # filter the ref_condition_list based on the drug
        current_drug <- celllines[grep(condition,celllines)]
        sub_lists[[condition]] <- current_drug
    }
    for(i in names(sub_lists)){
      cat(paste0("\n\n## Odds Ratio Comparing Class + drug: ", i))
      tmp = sub_lists[[i]]
      for(class in classes){
        reference_condition <-  class
        comparing_conditions <- setdiff(unique(classes), class)
        for(comparing_condition in comparing_conditions){
        
          #only look at data within the same drug
          matching_fit_data <- fit_data[fit_data$CellLine %in% tmp, ]
          matching_fit_data$Drug = factor(matching_fit_data$Drug)
        
          cat(paste0("\n\n## Odds Ratio Comparing: ", reference_condition, " vs ", comparing_condition," within drug ", i, "\n\n"))
          mrid_lmer_stats(matching_fit_data, reference_condition, comparing_condition, output_path, 
                        compare_cellline, compare_class, compare_drug, 
                        multiple_experiments, multiple_plates, TimepointIsNumeric)
        }
      }
    }
  
  }
  else if(compare_class){
    #compare class within the same drug
    #compare_cellline <- FALSE, compare_class <- TRUE, compare_drug <- FALSE
    classes <- levels(factor(data$Class))
  
    cat("\n\n## Odds Ratio Comparing Class: ")
    for(class in classes){
      reference_condition <-  class
      comparing_conditions <- setdiff(unique(classes), class)
      for(comparing_condition in comparing_conditions){
        
        #only look at data within the same drug
        matching_data <- data[data$Class %in% reference_condition | data$Class %in% comparing_condition, ]
        
        cat(paste0("\n\n## Odds Ratio Comparing: ", reference_condition, " vs ", comparing_condition," with no drug \n\n"))
        mrid_lmer_stats(matching_data, reference_condition, comparing_condition, output_path, 
                        compare_cellline, compare_class, compare_drug, 
                        multiple_experiments, multiple_plates, TimepointIsNumeric)
      }
    }
  }
  #same class - different drug
  else if(compare_cellline && compare_drug) {
    #compare cross drugs within same class(combined celllines): 
    #compare_cellline <- TRUE, compare_class <- FALSE, compare_drug <- TRUE

    for (class in unique(classes_string)) {
      
      cat(paste0("\n\n## Analyzing Class: ", class, "\n\n"))
      
      # Filter data for the current class
      class_data <- subset(fit_data, grepl(class, CellLine))
      class_data$Drug <- paste(class_data$Drug, class, sep = ' ')
      
      for (drug in unique(class_data$Drug)) {
        reference_condition <- drug
        comparing_conditions <- setdiff(unique(class_data$Drug), drug)
          
        for(j in 1:length(comparing_conditions)){
          comparing_condition = comparing_conditions[j]
            
          # Filter data for the two drugs within the class
          matching_fit_data <- subset(class_data, Drug %in% c(reference_condition, comparing_condition))
          
          # Combining cell lines within each drug for comparison
          cat(paste0("\n### Odds Ratio Comparing: ", reference_condition, " vs ", comparing_condition, " within Class '", class, "'\n"))
          mrid_lmer_stats(matching_fit_data, reference_condition, comparing_condition, output_path, 
                                   compare_cellline, compare_class, compare_drug, 
                                   multiple_experiments, multiple_plates, TimepointIsNumeric)
        }
      }
    }
  }
}


mrid_logodd_combine_plot <- function(data, reference_condition,output_path, colorpalette, confident_int,compare_class=FALSE) {
  
  data <- data[data$Reference==reference_condition,] 
  data$Compare <- factor(data$Compare)
  
  write.csv(data,file.path(output_path,paste0("Combined_OR_",reference_condition,".csv")))
  
  split_tibble <- function(tibble, col = 'col') tibble %>% split(., .[,col])
  dflist <- split_tibble(data, 'Compare')
  
  if("Drug" %in% names(data)){
    p1 <- ggplot(data, aes(y=OR, x=Time, fill = Drug))+
      theme_bw()+ 
      geom_hline(yintercept = 1, linetype=2) + 
      geom_line(stat="identity",aes(color =  Drug))+
      scale_x_continuous("Time (in days)", labels = as.character(data$Time), breaks = data$Time)+
      ggtitle(paste0("Odd Ratio - Reference: ",reference_condition))
    
    p2 <- ggplot(data, aes(y=OR, x=Time, fill = Drug))+
      theme_bw()+
      geom_hline(yintercept = 1, linetype=2) + 
      geom_line(stat="identity",aes(color = Drug))
  }else{

    p1 <- ggplot(data, aes(y=OR, x=Time, fill = Compare))+
      theme_bw()+ 
      geom_hline(yintercept = 1, linetype=2) + 
      geom_line(stat="identity",aes(color =  Compare))+
      scale_x_continuous("Time (in days)", labels = as.character(data$Time), breaks = data$Time)+
      ggtitle(paste0("Odd Ratio - Reference: ",reference_condition))
    
    if(!is.null(colorpalette)){
      names(colorpalette) <- gsub("[/_]"," ",names(colorpalette))
      p1 <- p1 + scale_color_manual(values=colorpalette)
    }
    
    #Odd Ratio in 95 or 99% confident interval
    p2 <- ggplot(data, aes(y=OR, x=Time, fill = Compare))+
      theme_bw()+
      geom_hline(yintercept = 1, linetype=2) + 
      geom_line(stat="identity",aes(color =  Compare))
  }
  
  for(i in 1:length(dflist)) {
    df <- do.call(rbind.data.frame, dflist[i])
    if (confident_int == 0.95) {
      p2 <- p2 + geom_ribbon(data=df, aes(ymin = lower, ymax = upper), alpha = 0.2, show.legend = FALSE)
    } else if (confident_int == 0.99) {
      p2 <- p2 + geom_ribbon(data=df, aes(ymin = lower99, ymax = upper99), alpha = 0.2, show.legend = FALSE)
    }
  }
  p2 <- p2 + 
    scale_x_continuous("Time (in days)", labels = as.character(data$Time), breaks = data$Time) +
    ggtitle(paste0("Odd Ratio ", confident_int * 100, "% C.I. - Reference: ", reference_condition)) 
  
  p3 <- p2 +
    facet_wrap(~Compare, scales = "free")
  
  print(p1)
  cat("\n\n")
  print(p2)
  cat("\n\n\\pagebreak\n")
}
