function var = distortInitialValues(start,stop,var)
var = start*var + (stop*var-start*var).*rand(size(var));
end