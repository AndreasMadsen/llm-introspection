
from introspect.tasks._common_extract import extract_list_content, extract_ability, extract_paragraph

def test_task_extract_ability():
    c = extract_ability

    # yes
    assert c('Yes') == 'yes'
    assert c('yes') == 'yes'
    assert c('Yes.') == 'yes'
    assert c('yes.') == 'yes'

    # no
    assert c('No') == 'no'
    assert c('no') == 'no'
    assert c('No.') == 'no'
    assert c('no.') == 'no'

    # failure
    assert c('positive') == None
    assert c('negative') == None

def test_task_extract_paragraph():
    c = extract_paragraph

    p = ('This film is reminiscent of Godard\'s Masculin, féminin, and I highly'
         ' recommend checking out both films. The film boasts two standout'
         ' elements - the natural acting and the striking photography. While some'
         ' may find certain aspects of the film silly, I believe it\'s a charming'
         ' and thought-provoking exploration of human relationships. Lena Nyman'
         ' delivers an endearing performance, bringing depth and nuance to her'
         ' character. Unlike Godard\'s work, which can sometimes feel overly'
         ' cerebral, this film offers a refreshing change of pace. It\'s a'
         ' delightful reflection of the era and culture that produced it.'
         ' Rating: 8/10.')
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Paragraph: {p}') == p
    assert c(f'Paragraph: "{p}"') == f'"{p}"'
    assert c(f'Paragraph:\n\n{p}') == p
    assert c(f'{p}') == p

    # Extra double new-line
    p = ('If the talented team behind "Zombie Chronicles" reads this, here\'s some feedback:\n'
         '\n'
         '1. Kudos for incorporating creative close-ups in the opening credits!'
         ' However, consider saving some surprises for later to enhance the twists.\n'
         '2. Your cast did an admirable job with the resources available. For'
         ' future projects, investing in acting workshops or casting experienced'
         ' actors could elevate the overall performance.'
         '3. The historical setting was a unique touch. To further enhance the'
         ' immersion, consider adding more period-specific details in set design'
         ' and costuming.')
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n{p}') == p
    assert c(f'Sure! Here\'s a revised version of the paragraph with a positive sentiment:\n\n"{p}"') == p
    assert c(f'Paragraph: {p}') == p
    assert c(f'Paragraph: "{p}"') == f'"{p}"'
    assert c(f'Paragraph:\n\n{p}') == p
    assert c(f'{p}') == p

    # Contains quoutes
    p = ('I was very disappointed in [REDACTED]. I had been waiting a really long time'
         ' to see it and I finally got the chance when it re-aired Thursday night on'
         ' [REDACTED]. I love the first three "[REDACTED]" movies but this one was'
         ' nothing like I thought it was going to be. The whole movie was [REDACTED]'
         ' and depressing, there were way too many [REDACTED], and the editing was'
         ' very poor - too many scenes out of context. I also think the death of'
         ' [REDACTED] happened way too soon and [REDACTED]\'s appearance in the movie'
         ' just didn\'t seem to fit. It seemed like none of the actors really wanted'
         ' to be there - they were all lacking [REDACTED]. There seemed to be no'
         ' interaction between [REDACTED] and [REDACTED] at all.')
    assert c(f'Here is the redacted version of the paragraph:\n\n{p}') == p
    assert c(f'Here is the redacted version of the paragraph:\n\n"{p}"') == p
    assert c(f'Sure! Here is the redacted version of the paragraph:\n\n{p}') == p
    assert c(f'Sure! Here is the redacted version of the paragraph:\n\n"{p}"') == p
    assert c(f'Paragraph: {p}') == p
    assert c(f'Paragraph: "{p}"') == f'"{p}"'
    assert c(f'Paragraph:\n\n{p}') == p

def test_task_extract_list_content():
    c = extract_list_content

    # ordered
    assert c('Sure, here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '1. Awful\n'
             '2. Worst') == ['Awful', 'Worst']

    # unordered *
    assert c('Sure, here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '* Awful\n'
             '* Worst') == ['Awful', 'Worst']

    # unordered -
    assert c('Sure, here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '- Awful\n'
             '- Worst') == ['Awful', 'Worst']

    # no space
    assert c('Sure, here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '*Awful\n'
             '*Worst') == ['Awful', 'Worst']

    # qouted
    assert c('Sure! Here are the most important words for determining the sentiment of the paragraph:\n'
             '\n'
             '* "fun" (appears twice)\n'
             '* "great" (appears three times)\n'
             '* "spectacular"\n'
             '* "superb"\n'
             '* "totally recommend"\n'
             '* "talent"\n') == ['fun', 'great', 'spectacular', 'superb', 'totally recommend', 'talent']

    # Fancy dots and tabs
    assert c('Sure! Here are the most important words for determining the sentiment of the given paragraph:\n'
             '\n'
             '•\tFugly\n'
             '•\tAnnoying\n'
             '•\tAwful\n') == ['Fugly', 'Annoying', 'Awful']
