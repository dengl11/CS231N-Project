def parse_img_index(s):
    """
    parse image index:  dasheng-6298.jpg -> 6298
    """
    return int(s.split("-")[1].split(".")[0])