import ast
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import pytest
from language_keys import resolve_dataset_languages, translation_texts


@pytest.mark.parametrize('source,target,expected', [('en','yo',('en','yo')), ('en_XX','fr_XX',('en','fr')), ('eng_Latn','yor_Latn',('eng','yor'))])
def test_legacy_defaults(source,target,expected):
    assert resolve_dataset_languages(source,target) == expected


def test_bambara_proxy_model_language_is_separate():
    source, target = resolve_dataset_languages('fr','sw','fr','bam')
    assert translation_texts({'translation':[{'fr':'bonjour','bam':'i ni ce'}]},source,target) == (['bonjour'],['i ni ce'])


@pytest.mark.parametrize('batch', [{}, {'translation':[{'fr':'x'}]}, {'translation':[{'fr':'x','bam':None}]}, {'translation':[42]}])
def test_invalid_dataset_fields(batch):
    with pytest.raises(ValueError): translation_texts(batch,'fr','bam')


def test_empty_key_does_not_silently_default():
    with pytest.raises(ValueError): resolve_dataset_languages('fr','sw','fr','')


def test_actual_preprocess_function_uses_dataset_keys():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'run_translation.py').read_text())
    func = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='preprocess_function')
    class Tokenizer:
        def __init__(self): self.calls=[]
        def __call__(self,texts,**kwargs): self.calls.append(texts); return {'input_ids':[[1] for _ in texts]}
        def as_target_tokenizer(self): return nullcontext()
    tokenizer=Tokenizer()
    ns=dict(translation_texts=translation_texts,source_lang='fr',target_lang='bam',prefix='',tokenizer=tokenizer,
            data_args=SimpleNamespace(max_source_length=20,ignore_pad_token_for_loss=True),padding=False,max_target_length=20)
    exec(compile(ast.Module(body=[func],type_ignores=[]),'<actual-preprocess>','exec'),ns)
    result=ns['preprocess_function']({'translation':[{'fr':'bonjour','bam':'i ni ce'}]})
    assert tokenizer.calls == [['bonjour'], ['i ni ce']]
    assert result['labels'] == [[1]]
