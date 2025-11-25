# ai-training-
training local AI to parse data

the only true requirements to run this is to have your own API keys for the following
    OPENAI API KEY,
    SUBABASE ACCOUNT,
    SUPABASE URL,
    SUPABASE API KEY,

the run_pipline.py script will run you through everything all at once, there are manual inputs and manual steps to take along the way

if you want to, below are the stpes to run each script in order to see whats its doing

step 1: 
    run the generic_csv_cleaner.py script first to get the csv in the correct format, you may lose your column headers depending on its format

step 2: 
    run the modify_csv_testing.py script to grab the headers and put it in a text file called headers_output.txt 
    each time you run the modify csv script it will override the previous contents of the headers txt file so be mindful.
    you only have to do the sql setup once for supabase per table so its no biggie if the headers get overriden after you have created the table

step 3:
    run the header_modify_for_sql.py script
    manually add in the last column 'embedding vector(1536)'

step 4: 
    run the rearrange_date.py script to change the column headers in the csv as well as the modified sql text document
    take the contents of header_modifited_for_sql and paste them into the sql editor to create the table and column headers

step 5:
    run the csv_import_to_supabase.py

step 6: 
    run the generate_rpc_function.py script to generate a 2nd query for supabase that you will add to query the table  in the terminal  

step 7:
    you can run the query_supabase_generic.py script to ask questions about the database