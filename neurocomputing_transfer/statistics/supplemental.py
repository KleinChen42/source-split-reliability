"""Pre-outcome one-factor-at-a-time catalog, with fixed candidate ordering."""

def variant_catalog(plan,dataset):
    settings=plan.get('supplemental',{})
    if not settings.get('enabled',False):return []
    seeds=settings['split_seeds']
    variants=[]
    def add(name,**kwargs):variants.append({'id':name,'split_seeds':seeds,**kwargs})
    add('native_only',scorer='native_only')
    add('constant_only',scorer='constant_only')
    add('common_only',scorer='common_only')
    add('common_plus_native',features='common_plus_native')
    add('text_only',features='text_only')
    add('visual_only',features='visual_only')
    for fraction in [.25,.5,1.]:add(f'matched_fit_{fraction}',matched_fit_fraction=fraction)
    for cap in [2,5]:add(f'holdout_cap_{cap}',holdout_record_cap=cap)
    if dataset=='docvqa':add('page_instead_of_parent',source_unit='page',flows=['source_disjoint'])
    for q in plan['selection']['secondary_coverages']:add(f'coverage_{q}',coverage=q)
    panel=plan['panels']['video' if dataset=='activitynetqa' else 'image_document']
    total=len(panel)*len(plan['prompts'])
    for n in plan['stress']['candidate_counts']:
        if n<total:add(f'candidates_{n}',candidate_count=n)
    for alpha in plan['stress']['overlap_fraction']:add(f'overlap_{alpha}',overlap_fraction=alpha,flows=['controlled_overlap'])
    for mode in plan['stress']['source_id_error_modes']:
        for fraction in plan['stress']['source_id_error_fraction']:
            add(f'source_{mode}_{fraction}',source_error_mode=mode,source_error_fraction=fraction,flows=['source_disjoint'])
    return variants
