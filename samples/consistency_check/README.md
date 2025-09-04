In this folder there is the code used to ensure that this implementation is consistent with the original implementation of [smplx](https://github.com/vchoutas/smplx).

Procedure:
- Run both `original_api.py` and `./consistency_check`, this will generate two output files: `output_py.txt` and `output_cpp.txt`
- Run `analysis.py` to compare the results
    ```
    python3 analysis.py ../../build/output_cpp.txt output_py.txt 
    ```