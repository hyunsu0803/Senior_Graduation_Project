from bvh_handler import parsing_bvh
def drawTpose():
    paths = []
    paths.append('./lafan1/fallAndGetUp1_subject1.bvh')

    with open(paths[0], 'r') as file:
        FPS = parsing_bvh(file)
        