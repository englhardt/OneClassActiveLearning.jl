using PyCall
const PACKAGES = ["scipy"]

try
    pip = pyimport("pip")
catch
    get_pip = joinpath(dirname(@__FILE__), "get-pip.py")
    download("https://bootstrap.pypa.io/3.5/get-pip.py", get_pip)
    run(`$(PyCall.python) $get_pip --user`)
end

run(`$(PyCall.python) -m pip install --user --upgrade pip setuptools`)
run(`$(PyCall.python) -m pip install --user $(PACKAGES)`)
