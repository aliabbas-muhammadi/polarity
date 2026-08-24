from polarity.data.categorize import categorize
from polarity.types import Category, Label

E = Label.CACHE_EQUIVALENT
N = Label.NOT_CACHE_EQUIVALENT


def test_exact_repeat():
    assert categorize("What is X?", "what is x?", E) == Category.EXACT_REPEAT


def test_equivalent_negation_cue():
    assert categorize("This is not secure.", "This is risky.", E) == Category.EQUIVALENT_NEGATION


def test_equivalent_negation_morphology():
    assert categorize("The build is stable.", "The build is unstable.", E) == (
        Category.EQUIVALENT_NEGATION
    )


def test_plain_paraphrase_is_not_equivalent_negation():
    cat = categorize("Who wrote Dune?", "Who wrote the book Dune?", E)
    assert cat in {Category.PARAPHRASE, Category.HARD_PARAPHRASE}


def test_explicit_negation():
    assert categorize("Is it deprecated?", "Is it not deprecated?", N) == Category.EXPLICIT_NEGATION


def test_lexical_contradiction():
    assert categorize("Is it safe?", "Is it unsafe?", N) == Category.LEXICAL_CONTRADICTION


def test_numeric_change():
    assert categorize("What is 15% of 200?", "What is 15% of 400?", N) == Category.NUMERIC_CHANGE


def test_scope_change():
    assert categorize("Dose for adults?", "Dose for infants?", N) == Category.SCOPE_CHANGE


def test_quantifier_change():
    assert categorize("Do all birds fly?", "Do some birds fly?", N) == Category.QUANTIFIER_CHANGE


def test_unknown_when_no_signal():
    assert categorize("How do I reset my password?", "How do I change my username?", N) == (
        Category.UNKNOWN
    )
