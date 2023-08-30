from sql import Database



def delete_experiment(opt):
    Db = Database()
    exp_dct = dict(experiment=opt.experiment,
                )
    Db.delete_based_on_duplicate_name('experimentdata', exp_dct)
