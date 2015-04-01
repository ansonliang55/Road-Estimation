vers = version;
mx = mexext;
if ispc
  mx = 'dll';  % Use dll extension for both Matlab 6 and 7
end

eval(['mex slicmex.c -output slicmex.' mx]);
