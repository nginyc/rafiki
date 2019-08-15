import React from 'react';
import { withStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import Typography from '@material-ui/core/Typography';

import SyntaxHighlighter from 'react-syntax-highlighter';
import { gruvboxDark, solarizedLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';


const styles = {
  card: {
    maxWidth: "100%",
  },
};

function DocsCard(props) {
  const { classes } = props;
  return (
    <Card className={classes.card}>
      <CardContent>
        <Typography gutterBottom variant="h3" component="h1">
          Put Data Entry by CSV
        </Typography>
        <Typography component="p">
          Put data entries to an existing dataset by uploading a CSV file
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'PUT_DATA_ENTRY_BY_CSV​ <file> -t <dataset> -b <branch> -m <entry_name_indices>'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {`// path to the file on system:\nfile-path`}
        </SyntaxHighlighter>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {`// the operating table or dataset:\n-t [ --table ] arg`}
        </SyntaxHighlighter>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {`// the operating branch:\n-b [ --branch ] arg`}
        </SyntaxHighlighter>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {`
// the operating column or data entry
// for datasets, entry_name_indices(-m) indicates the row for the schema
// if --with-schema is not provided the first row is treated as normal data
-m [ --column ] arg`}
        </SyntaxHighlighter>
        <Typography component="p">
          Utility Options:
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'--with-schema  // input data containing schema at the 1st line'}
        </SyntaxHighlighter>
        <br />

        <Typography variant="h5" gutterBottom>
          Examples
        </Typography>
        <SyntaxHighlighter language='javascript' style={gruvboxDark}>
          {`
ustore> put_data_entry_by_csv ../../../mock-data/sample.csv -t sampleDS1 -b master -m 0 --with-schema
[SUCCESS: PUT_DATA_ENTRY_BY_CSV] 5 entries are updated  [140B]

// Note same file uploaded, --with-schema will treat the first row as schema
ustore> put_data_entry_by_csv ../../../mock-data/sample.csv -t DS2 -b master -m 0
[SUCCESS: PUT_DATA_ENTRY_BY_CSV] 6 entries are updated (no schema)  [143B]
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
