# Academic Profile Analysis
This project is to do a comparison and breakdown of academic profiles and their careers from NTU faculties and NUS Computing.

All the data used were obtained manually from scratch through scraping using academic profiles into "FINAL DATA.csv" and "nus profiles.csv" for NTU and NUS respectively through BeautifulSoup in Python. For each profile, their respective details (e.g. h_index, key topics with each profile, etc) was obtained through Google Scholar page.

Insights such as citations growth rate, career length were derived through the other scraped columns from these websites.

A streamlit website was deployed to visualise these insights, as well as do comparisons against SCSE (NTU) and other faculties (from NTU and NUS).
To view website: https://kirubhaharini-data-products-assgn-2-assgn2-4kvt4t.streamlit.app/
*Note: Website was built using older streamlit version. Some elements may have depreciated.

The features of the website are as follows:

1. SCSE profile analysis:
   
   <img width="374" height="312" alt="image" src="https://github.com/user-attachments/assets/635b6d51-4c54-46c7-bd25-d0641a4ae5d0" />

   a) Distribution of profiles
   
    <img width="700" height="500" alt="image" src="https://github.com/user-attachments/assets/650719ab-63ed-47e6-8252-5cac1297ebb4" />
   b) Performance in top research areas
    <img width="1780" height="812" alt="image" src="https://github.com/user-attachments/assets/b163ec6d-f5f7-4b60-be06-0414488df200" /> <img width="1648" height="812" alt="image" src="https://github.com/user-attachments/assets/db9c377c-fce5-4f80-a7e3-7cfb6b89db3e" />
   c) Comparisons of SCSE agianst other faculties
    <img width="2504" height="1106" alt="image" src="https://github.com/user-attachments/assets/156d0c58-990e-48ec-ab7e-7b877e7a34a8" />

3. Compare Individual Researchers



   <img width="406" height="384" alt="image" src="https://github.com/user-attachments/assets/fd10ebd4-2622-4ef6-9693-2a2ff1e7e9ce" />

   Three comparison options:
   (a) "Within SCSE": Any 2 researchers from SCSE
   (b) "Within NTU": Any SCSE researcher with researcher from another faculty
   (c) "With NUS": Any NTU researcher with researcher from NUS Computing

   Comparisons will be shown side by side with insights such as Career Length, number of papers as lead author, citations by year, common interests, top 5 co-authors, average h-index percentile within their respective faculties and top 5 publication venues.

   <img width="2248" height="1054" alt="image" src="https://github.com/user-attachments/assets/c6f801e3-d5f1-4fc5-9c2a-b4f21982c747" />

   <img width="2312" height="1334" alt="image" src="https://github.com/user-attachments/assets/47f804b4-e6b8-4b89-9692-3a184a37658c" />





* NTU data obtained from https://dr.ntu.edu.sg/explore/researcherprofiles
* NUS data obtained from https://www.comp.nus.edu.sg/about/faculty/
