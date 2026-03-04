class PaperSchema(sl.Schema):
    text: sl.String
    published: sl.Timestamp  # This will handle datetime objects properly
    entry_id: sl.IdField
    title: sl.String
    summary: sl.String


paper = PaperSchema()
