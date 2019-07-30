from absl import flags
from absl import app
from absl import logging

import os
import time
import torch

import sys
sys.path.append('/trinity/home/y.gusak/musco/')
from tensor_compression import get_compressed_model

import sys
sys.path.append('/trinity/home/y.gusak/ReducedOrderNet/src')
from models.dcp.pruned_resnet import PrunedResNet

FLAGS = flags.FLAGS


flags.DEFINE_integer("global_iter", None, "Global iter (how many times do we compress the whole model?)")
flags.DEFINE_integer("local_iter", None, "Local iter (which part of model we compress during global iter)")
flags.DEFINE_integer("max_local_iter", None, "Maximum available local iter (which part of model we compress the last during global iter)")


flags.DEFINE_string("global_dir", None, "Directory to save compressed models of one compression schedule")
flags.DEFINE_string("initialmodel_path", None, "Path to the initial (uncompressed) model")
flags.DEFINE_string("initialstate_path", None, "Path to the initial (uncompressed) model state")

flags.DEFINE_string("decomposition", "tucker2", "Decomposition algorithm. Available tucker2 and cp3")
flags.DEFINE_string("rank_selection", "vbmf", "Algorithm for rang calculation.")
flags.DEFINE_float("factor", None, "Weakenen factor if --rank_selection vbmf, Reduction factor if --rank_selection nx")

flags.DEFINE_spaceseplist("lnames", None, "Names of layers to be compressed")


def make_savedir():
    savedir = "{}/iter_{}-{}".format(FLAGS.global_dir,\
                                     FLAGS.global_iter,\
                                     FLAGS.local_iter)
#     make_catalog(savedir)
    return savedir


def make_logsdir(savedir):
    logsdir = "{}/comprlogs".format(savedir)
    make_catalog(logsdir)
    return logsdir
    
    
def make_catalog(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_prevmodelpath():
    if FLAGS.global_iter == 0 and FLAGS.local_iter == 0:
        prevmodelpath = FLAGS.initialmodel_path
        prevstatepath = FLAGS.initialstate_path
    elif FLAGS.local_iter == 0:
        #prevmodelpath = "{}/iter_{}-{}/best.pth".format(FLAGS.global_dir,\
        #                                                    FLAGS.global_iter-1,\
        #                                                    FLAGS.local_iter)
        prevmodelpath = "{}/iter_{}-{}/beforeft.pth".format(FLAGS.global_dir,\
                                                            FLAGS.global_iter-1,\
                                                            FLAGS.max_local_iter)
        prevstatepath = "{}/iter_{}-{}/best.pth.tar".format(FLAGS.global_dir,\
                                                            FLAGS.global_iter-1,\
                                                            FLAGS.max_local_iter)


    else:
        #prevmodelpath = "{}/iter_{}-{}/best.pth".format(FLAGS.global_dir,\
        #                                                    FLAGS.global_iter,\
        #                                                    FLAGS.local_iter-1)
        prevmodelpath = "{}/iter_{}-{}/beforeft.pth".format(FLAGS.global_dir,\
                                                            FLAGS.global_iter,\
                                                            FLAGS.local_iter-1)
        prevstatepath = "{}/iter_{}-{}/best.pth.tar".format(FLAGS.global_dir,\
                                                            FLAGS.global_iter,\
                                                            FLAGS.local_iter-1)

    return prevmodelpath, prevstatepath
        

def make_modelpath(savedir):
    return "{}/beforeft.pth".format(savedir)


    
        

def main(_):
    savedir = make_savedir()
    logsdir = make_logsdir(savedir)
    
    logging.get_absl_handler().use_absl_log_file("compress_logs.log",\
                                                 log_dir = logsdir)
       
    logging.info("Global iter: {}, local iter: {}, decomposition: {}, rank_selection: {}, factor: {}".format(FLAGS.global_iter, FLAGS.local_iter, FLAGS.decomposition, 
         FLAGS.rank_selection, FLAGS.factor))
        
    modelpath = make_modelpath(savedir)
    prevmodelpath, prevstatepath = get_prevmodelpath()

    logging.info("Compress model {} with state {}".format(prevmodelpath, prevstatepath))
    logging.info("Compress layers: {}".format(FLAGS.lnames))
    logging.info("Save model: {}".format(modelpath))
    
    
    if not os.path.isfile(prevstatepath) or not os.path.isfile(prevmodelpath):
        logging.info("No model to compress")
    else:
        logging.info("Start compression")
            
    
    if FLAGS.rank_selection == "vbmf":
        rank = 0
        wf = FLAGS.factor
    elif FLAGS.rank_selection == "nx":
        rank = -FLAGS.factor
        wf = None
    
    n_lnames = len(FLAGS.lnames)
    ranks = [rank]*n_lnames
    decompositions = [FLAGS.decomposition]*n_lnames
    
    
    if not os.path.exists(modelpath):

        start = time.time()
        compressed_model = torch.load(prevmodelpath, map_location = 'cpu')
        
        if not(FLAGS.global_iter == 0 and FLAGS.local_iter == 0):
            compressed_model.load_state_dict(torch.load(prevstatepath, map_location = 'cpu')['state_dict']) 
#         except:
#             compressed_model.load_state_dict(torch.load(prevstatepath, map_location = 'cpu'))
        
        compressed_model = get_compressed_model(compressed_model,
                                            ranks=ranks,
                                            layer_names=FLAGS.lnames,
                                            decompositions = decompositions,
                                            vbmf_weaken_factor= wf)
        torch.save(compressed_model, modelpath)   

        end = time.time()
        logging.info("End compression, total time: {}".format(end - start))  


if __name__=="__main__":
    app.run(main)
