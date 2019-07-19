import os, sys, importlib
from termcolor import colored

try:
    f = sys.argv[1]
    if '.py' in f: f = f[:-3]
    script = 'scripts.' + f

    print('🐉  Running script ' + colored(f, 'cyan'))
    importlib.import_module(script)
    print('🦄  It\'s done, my dude\n')

except (IndexError, ModuleNotFoundError):
    com = colored('python train.py {script_name}', 'yellow')
    print('\nYou\'re an idiot ❤️')
    print('You need to specify which script to execute as so:\n' + com)

    scripts = [f[:-3] for f in os.listdir('scripts') if '.py' in f]
    print('\n🍺  ' + colored('Available Scripts:', 'cyan'))
    for script in scripts:
        print(' + ' + script)

except KeyboardInterrupt:
    com = colored('Cancelled', 'red')
    sys.stdout.write(f'\r🦀  Oh no it\'s the cancel crab\n{com}\n')

except:
    raise
