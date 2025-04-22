###########################
## ---- RUN COMMAND
## Rscript --vanilla run_compete.R --data_str=GGM1 --data_seed=0 --train_size=10000
##
## ---- ADDITIONAL NOTES FOR CONDA ENV SETUP
## conda create -n rvenv
## conda activate rvenv
## conda install r-base
## conda install r-essentials
## conda install numpy
## note that doing directly `conda create -n rvenv r-base r-essentials may get stuck`
###########################
### SETUP LIBRARIES
rm(list=ls())
pkgs = c("reticulate", "abind", "huge", "yaml", "optparse");
new = pkgs[!(pkgs%in%installed.packages()[,"Package"])]
if (length(new))
{
    for (pkg in new)
    {
        install.packages(pkg, dependencies = TRUE, repos='http://cran.us.r-project.org')
    }
}
sapply(pkgs, require, character.only = TRUE);

## HELPER FUNCTION FOR LOGGING
logger = function(msg, LEVEL='INFO')
{
    cat(sprintf("%s;%s - %s\n", format(Sys.time(), "%Y-%m-%d %X"), LEVEL, msg))
}

## MAIN FUNCTION THAT LOOPS THROUGH EACH METHOD
main = function(){
    
    ## process config information
    configs = read_yaml(sprintf("./configs/competitors/%s_r.yaml", args$data_str))

    run = tryCatch({
    
        ## getting ready to load npy files
        logger('Setting up numpy for loading and saving data')
        stopifnot(!(is.null(py_config()$numpy$path)))
        np = reticulate::import("numpy")

        ## read data
        logger(sprintf('Reading data for %s', args$data_str))
        data_folder = sprintf('data_sim/%s_seed%d', args$data_str, args$data_seed)
        data_full = np$load(sprintf("%s/x_train.npy", data_folder))
        regime = np$load(sprintf("%s/regime_train.npy", data_folder))
        
        if (args$regime == 0)
        {
            x = data_full[1:min(args$train_size, nrow(data_full)),]
        }
        else
        {
            valid_indices = which(regime==args$regime)
            valid_indices = valid_indices[valid_indices<=args$train_size]
            x = data_full[valid_indices,]
        }
        
        n = nrow(x)
        p = ncol(x)
        logger(sprintf('data loaded;  n=%d, p=%d', n, p))

        ## set up output folder
        output_folder = file.path(sprintf('output_%s', args$run_type), 'competitors')

        ## estimate using neighborhood selection
        logger('Conducting estimation using neighborhood selection (mb)')
        if (is.null(configs$mb$lambda_low) | is.null(configs$mb$lambda_high)){
            lambda_seq = NULL
            nlambda = configs$mb$lambda_length
        }
        else {
            lambda_seq = sqrt(log(p)/n) * exp(seq(log(configs$mb$lambda_low), log(configs$mb$lambda_high), length.out=configs$mb$lambda_length))
            nlambda = configs$mb$lambda_length
        }
        huge_mb = huge::huge(x = x,
                             lambda = lambda_seq,
                             nlambda = nlambda,
                             method = 'mb',
                             sym = "or",
                             verbose = FALSE)
        out_mb = abind(lapply(huge_mb$path, as.matrix), along = 3)
        if (args$regime==0)
        {
            output_path_mb = file.path(output_folder, sprintf('%s_seed%d_mb.npy', args$data_str, args$data_seed))
        }
        else
        {
            output_path_mb = file.path(output_folder, sprintf('%s_seed%d_mb_regime%d.npy', args$data_str, args$data_seed, args$regime))
        }
        
        reticulate::r_to_py(out_mb)$dump(output_path_mb)
        logger(sprintf('mb result saved to %s', output_path_mb))
        
        ## estimate using glasso
        logger('Conducting estimation using glasso')
        if (is.null(configs$glasso$lambda_low) | is.null(configs$glasso$lambda_high)){
            lambda_seq = NULL
            nlambda = configs$glasso$lambda_length
        }
        else {
            lambda_seq = sqrt(log(p)/n) * exp(seq(log(configs$glasso$lambda_low), log(configs$glasso$lambda_high), length.out=configs$glasso$lambda_length))
            nlambda = configs$glasso$lambda_length
        }
        
        huge_glasso = huge::huge(x = x,
                                 lambda = lambda_seq,
                                 nlambda = nlambda, 
                                 method = 'glasso',
                                 verbose = FALSE)
        out_glasso = abind(lapply(huge_glasso$icov, as.matrix), along = 3)
        if (args$regime==0)
        {
            output_path_glasso = file.path(output_folder, sprintf('%s_seed%d_glasso.npy', args$data_str, args$data_seed))
        }
        else
        {
            output_path_glasso = file.path(output_folder, sprintf('%s_seed%d_glasso_regime%d.npy', args$data_str, args$data_seed, args$regime))
        }
        reticulate::r_to_py(out_glasso)$dump(output_path_glasso)
        logger(sprintf('glasso result saved to %s', output_path_glasso))

        return(0)
        
    }, ## end of try
    error = function(err)
    {
        logger(err, LEVEL='ERROR')
        return(1)
    })
}

###################################
## main entry point
###################################
### SETUP CMD ARGS
option_list = list(
  make_option("--data_str", 
              type="character", 
              default=NULL),
  make_option("--data_seed", 
              type="integer", 
              default=0),
  make_option("--regime",
            type="integer",
            default=0),
  make_option("--train_size",
              type="integer", 
              default=10000,
              help="number of sample size"),
  make_option("--run_type", 
              type="character", 
              default='sim',
              help="run_type; sim or real")
);
args = parse_args(OptionParser(option_list=option_list));

### RECORD EXECUTION START TIME
dir.create(file.path(getwd(), sprintf('output_%s/competitors/', args$run_type)), recursive = TRUE)
ptm = proc.time()
### ACTUAL EXECUTION OF THE MAIN
exit_status = main()
logger(sprintf('Exit code=%d; time elapsed=%.2f seconds', exit_status, (proc.time() - ptm)[1]))
