import streamlit as st
import tfidf
#import bert

st.title('Поисковик по ответам mail.ru')

if 'is_expanded' not in st.session_state:
    st.session_state['is_expanded'] = True
container = st.expander('Поиск', expanded=st.session_state['is_expanded'])

with container:
    with st.form(key='myform', clear_on_submit=True):
        query = st.text_input('Запрос', placeholder='Поиск')
        search_type = st.selectbox('Тип поиска:',  ('TfIdfVectorizer', 'BERT'))
        submit_button = st.form_submit_button('Искать')

        st.session_state['is_expanded'] = False
if submit_button:
    if search_type == 'TfIdfVectorizer':
        st.header('Результаты поиска')
        results, time = tfidf.main(query)
        st.write(str(time))
        for i in range(len(results)-1):
            print(results[i])
            st.write(str(i+1), '. Вопрос: ', results[i]['question'])
            if type(results[i]['comment']) == str:
                st.write('Комментарий: ' + results[i]['comment'])
            if results[i]['answer']:
                st.write('Ответ: ' + results[i]['answer'])
    else:
        st.write('Берт в остался разработке')
    #    st.header('Результаты поиска')
    #    results, time = bert.main(query)
    #    st.write(str(time))
    #    for i in range(len(results)-1):
    #        print(results[i])
    #        st.write(str(i+1), '. Вопрос: ', results[i]['question'])
    #        if type(results[i]['comment']) == str:
    #            st.write('Комментарий: ' + results[i]['comment'])
    #        if results[i]['answer']:
    #            st.write('Ответ: ' + results[i]['answer'])

