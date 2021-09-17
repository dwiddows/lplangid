from lplangid import tokenizer


def test_tokenize_fast():
    assert tokenizer.tokenize_fast("Hello World!") == ["Hello", "World"]
    assert tokenizer.tokenize_fast("U.S.A.") == ["U.S.A"]
    assert tokenizer.tokenize_fast("U.S.A. means United States") == ["U.S.A", "means", "United", "States"]
    assert tokenizer.tokenize_fast("John's 'socks'") == ["John's", "socks"]
    assert tokenizer.tokenize_fast("Text with <br> HTML linebreak") == ["Text", "with", "HTML", "linebreak"]


def test_tokenize_url():
    assert tokenizer.tokenize_fast("http://go.microsoft.com/fwlink/?LinkID=267510&clcid=0x409") \
           == ["http://go.microsoft.com/fwlink/?LinkID=267510&clcid=0x409"]


def test_remove_html():
    assert tokenizer.remove_html_tags("Hi <br> there") == "Hi  there"
    assert tokenizer.remove_html_tags("Oh dear < This will all be removed >") == "Oh dear "
