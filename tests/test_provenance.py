from datetime import datetime

import pytest

from fusionbench import Provenance, __version__
from fusionbench.provenance import MODEL_REFERENCES, build_provenance


def test_build_provenance_fields():
    record = build_provenance(["bosch-hale-1992"], {"T": 10.0})
    assert record.package == "fusionbench"
    assert record.version == __version__
    assert record.models == ("bosch-hale-1992",)
    assert record.references == (MODEL_REFERENCES["bosch-hale-1992"],)
    datetime.fromisoformat(record.created)


def test_json_round_trip():
    record = build_provenance(["bosch-hale-1992", "brysk-1973"], {"T": 10.0})
    restored = Provenance.from_json(record.to_json())
    assert restored == record


def test_unknown_model_rejected():
    with pytest.raises(KeyError):
        build_provenance(["not-a-model"], {})
