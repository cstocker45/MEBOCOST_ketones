"""MEBOCOST: a python-based package to infer metabolite-mediated cell-cell communications in scRNA-seq data.
"""
import os
import setuptools
import configparser

def _make_config(conf_path, workdir=os.getcwd()):
    """
    read config file
    """
    #read config
    cf = configparser.ConfigParser()
    cf.read(conf_path)
    config = cf._sections
    # remove the annotation:
    for firstLevel in config.keys():
        for secondLevel in config[firstLevel]:
            if '#' in config[firstLevel][secondLevel]:
                path = config[firstLevel][secondLevel][:config[firstLevel][secondLevel].index('#')-1].rstrip()
                config[firstLevel][secondLevel] = os.path.join(workdir, path)
            else:
                path = config[firstLevel][secondLevel]
                config[firstLevel][secondLevel] = os.path.join(workdir, path)
    ## re-write
    cf_new = configparser.ConfigParser()
    for firstLevel in config.keys():
        cf_new.add_section(firstLevel)
        for secondLevel in config[firstLevel]:
            cf_new.set(firstLevel, secondLevel, config[firstLevel][secondLevel])
    with open('mebocost.conf', 'w') as f:
        cf.write(f)
    return(config)

## setup
def main():
  setuptools.setup(name="mebocost", 
                  version="1.2.2",
                  description="a python-based method to infer metabolite mediated cell-cell communication",
                  author='Rongbin Zheng, Kaifu Chen',
                  author_email='Rongbin.Zheng@childrens.harvard.edu',
                  url='https://kaifuchenlab.github.io/',
                  zip_safe=True,
                  package_dir={"": "src"},
                  packages=setuptools.find_packages(where="src"),
                  classifiers=[
                      'Environment::Console',
                      'Operating System:: POSIX',
                      "Topic :: Scientific/Engineering :: Bio-Informatics"],
                  keywords='Metabolism',
                  license='OTHER'
  )
if __name__ == '__main__':
    ## change mebocost.conf to absolute path
    _make_config(conf_path = './src/mebocost.conf')
    ## setup
    main()
