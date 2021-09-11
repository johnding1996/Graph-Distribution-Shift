# Graph Distribution Shift

## Overview
GDS (Graph Distribution Shift) is a benchmark for distribution shifts of graph data.

## Setup
To ensure that the developers are using exactly the same environment, please follow the guide below to set up the virtualenv.
1. Install Pyenv
Please follow the guide at https://github.com/pyenv/pyenv#installation; this does not require `sudo` privileges.
2. Install Python 3.7.9
~~~bash
$pyenv install 3.7.9

$pyenv rehash
~~~
3. Enter the project folder and check your Python version
~~~bash
$pyenv versions
  system
  3.6.15
  3.7.12
* 3.7.9 (set by /nfshomes/mcding/Graph-Distribution-Shift/.python-version)
  3.8.12
  3.9.7
  
$python -V
Python 3.7.9
~~~
By default, because of the `.python-version` config file, the python version you use will be automatically switched to `3.7.9`.

4. Create a virtualenv
~~~bash
$python -m venv env

$source env/bin/activate
~~~

6. Install the packages using the requirements.txt

If on Linux:
~~~bash
(env) $pip install -r requirements.linux.txt
~~~
If on Windows:
~~~bash
(env) $pip install -r requirements.win.txt
~~~
7. That's it. Please notify the team if you want to update some dependencies so that we can update the `requirements.txt`.
