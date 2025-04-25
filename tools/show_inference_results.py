import pickle

# load pkl file

pkl_file_path = "/home/rdluhu/Dokumente/ortho_test_images/sample_result/tile_3_rtdetr_detections.pkl"

with open(pkl_file_path, "rb") as f:
    result = pickle.load(f)


output = []
for i, prediction in enumerate(result.object_prediction_list):
    #print(i)
    x_min, y_min, x_max, y_max = prediction.bbox.to_xyxy()
    info = {
        "Index": i,
        "Category ID": prediction.category,
        "Score": prediction.score,
        "Bounding Box" : (x_min, y_min, x_max, y_max) 
    }
    output.append(info)

print("# Detection Results\n")
print("| Index | Category | Score | Bounding Box |")
print("|-------|-------------|-------|--------------|")
for info in output:
    print(f"| {info['Index']} | {info['Category ID']} | {info['Score']} | {info['Bounding Box']} |")


