from bvh_handler import parsing_bvh
def drawTpose():
    paths = []
    paths.append('./lafan1/aiming1_subject1.bvh')

    with open(paths[0], 'r') as file:
        FPS = parsing_bvh(file)
        file_name = (paths[0].split('/'))[-1].strip(".bvh")
    