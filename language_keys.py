"""Keep translation dataset keys independent of tokenizer language aliases."""


def resolve_dataset_languages(source_lang, target_lang, dataset_source_lang=None, dataset_target_lang=None):
    """Default to legacy short codes unless explicit dataset keys are supplied."""
    values = (source_lang, target_lang)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError('Source and target model language codes are required')
    source = dataset_source_lang if dataset_source_lang is not None else source_lang.split('_')[0]
    target = dataset_target_lang if dataset_target_lang is not None else target_lang.split('_')[0]
    if any(not isinstance(value, str) or not value.strip() for value in (source, target)):
        raise ValueError('Dataset language keys must be nonempty strings')
    return source, target


def translation_texts(examples, source_key, target_key):
    """Validate selected translation fields with diagnostics before tokenization."""
    if 'translation' not in examples:
        raise ValueError("Dataset is missing the 'translation' column")
    sources, targets = [], []
    for index, row in enumerate(examples['translation']):
        if not isinstance(row, dict):
            raise ValueError('Translation row {} must be an object'.format(index))
        missing = [key for key in (source_key, target_key) if key not in row]
        if missing:
            raise ValueError('Translation row {} is missing keys {}; available keys: {}. '
                'Set --dataset_source_lang/--dataset_target_lang independently of model codes.'.format(index, missing, sorted(row)))
        source, target = row[source_key], row[target_key]
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError('Translation row {} must contain source/target strings'.format(index))
        sources.append(source)
        targets.append(target)
    return sources, targets
