 # Let's extend from the base notebook
 FROM jupyter/base-notebook

 # Install required packages on top of base Jupyter image
 RUN pip install --no-cache-dir --upgrade pip \
   scipy \
   numpy \
   pandas 
#    scikit-learn \
#    matplotlib \
#    tensorflow

 # Copy all files (current directory onwards) into the image
 COPY . /