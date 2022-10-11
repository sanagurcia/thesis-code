from .. import ucrdata

def test_get_class_sequences():
    ADIAC_PATH = 'Adiac/Adiact_TEST.tsv'
    ADIAC_NO_CLASSES = 37
    data = ucrdata.get_class_sequences_2(10, ADIAC_PATH, ADIAC_NO_CLASSES)
    print(data)