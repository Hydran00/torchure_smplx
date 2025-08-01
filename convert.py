import numpy as np
import pickle
import sys, os, json
from os import path

if __name__ == "__main__":

    assert len(sys.argv) >= 2

    output_dir = os.path.curdir
    os.makedirs(output_dir, exist_ok=True)

    for model_path in sys.argv[1:]:
        with open(model_path, "rb") as model_file:
            try:
                model_data = pickle.load(model_file, encoding="latin1")
            except:
                model_data = pickle.load(model_file)

            output_data = {}
            for key, data in model_data.items():
                dtype = str(type(data))
                if "chumpy" in dtype:
                    output_data[key] = np.array(data).tolist()
                elif "scipy.sparse" in dtype:
                    output_data[key] = data.toarray().tolist()
                elif isinstance(data, np.ndarray):
                    output_data[key] = data.tolist()
                else:
                    try:
                        output_data[key] = np.array(data).tolist()
                    except Exception as e:
                        print(f"Skipping key {key} (unsupported type: {type(data)})")

            model_fname = path.split(model_path)[1]
            if len(model_fname) > 11 and model_fname[11] == "f":
                output_gen = "FEMALE"
            elif len(model_fname) > 11 and model_fname[11] == "m":
                output_gen = "MALE"
            else:
                output_gen = "NEUTRAL"

            output_path = path.join(output_dir, "SMPL_" + output_gen + ".json")
            print("Writing", output_path)
            with open(output_path, "w") as out_file:
                json.dump(output_data, out_file, indent=2)
