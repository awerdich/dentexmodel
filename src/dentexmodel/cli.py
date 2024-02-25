"""
Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

  You might be tempted to import things from __main__ later, but that will cause
  problems: the code will get executed twice:

  - When you run `python -m dentexmodel` python will execute
    ``__main__.py`` as a script. That means there won't be any
    ``final_project.__main__`` in ``sys.modules``.
  - When you import __main__ it will get executed again (as a module) because
    there's no ``dentexmodel.__main__`` in ``sys.modules``.

  Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import argparse
# from dentexmodel.script.create_dentex import main as create_dentex

parser = argparse.ArgumentParser(description='PyTorch Workflow')


# parser.add_argument('names', metavar='NAME', nargs=argparse.ZERO_OR_MORE, help="Model to be used.")
# parser.add_argument('-m', '--model', default = 'CNN_small', help = 'Model.')

def main(args=None):
    # We will add command line argument later.
    args = parser.parse_args(args=args)
    #create_dentex()
    print(f'Finished creating the Dentex data sets.')
