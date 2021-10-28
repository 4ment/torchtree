from torchtree.evolution.taxa import Taxa


def test_taxa():
    taxa = {
        "id": "taxa",
        "type": "torchtree.evolution.taxa.Taxa",
        "taxa": [
            {
                "id": "A_Belgium_2_1981",
                "type": "torchtree.evolution.taxa.Taxon",
                "attributes": {"date": 1981},
            },
            {
                "id": "A_ChristHospital_231_1982",
                "type": "torchtree.evolution.taxa.Taxon",
                "attributes": {"date": 1982},
            },
        ],
    }
    dic = {}
    taxa = Taxa.from_json(taxa, dic)
    assert len(taxa) == 2
    assert taxa[1].id == 'A_ChristHospital_231_1982'
    assert taxa[1]['date'] == 1982
