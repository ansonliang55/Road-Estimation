# install scipy image priocessing
brew install Homebrew/python/pillow

# install open CV python
brew tap homebrew/science
brew install opencv
brew install opencv --env=std
#vim ~/.bash_profile
#add> export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH