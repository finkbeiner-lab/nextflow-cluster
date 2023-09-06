## REFERENCE https://github.com/nturaga/bioc-galaxy-integration
## Command to run tool:
# Rscript Kmer_enumerate_tool.R --input Kmer_enumerate_test_input.fq --input 2 --output Kmer_enumerate_test_output.txt

# Set up R error handling to go to stderr

library("optparse")
 
option_list = list(
  make_option(c("-t", "--txt"), type="character", default="yosup", 
              help="dataset file name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default="out.txt", 
              help="output file name [default= %default]", metavar="character")
); 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$out)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
}
print(opt$txt)