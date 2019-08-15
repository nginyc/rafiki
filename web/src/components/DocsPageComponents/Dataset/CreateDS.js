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
          Create Dataset
        </Typography>
        <Typography component="p">
          Create a new empty dataset
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'CREATE_DATASET -t <dataset> -b <branch>'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {`// the operating table or dataset:\n-t [ --table ] arg`}
        </SyntaxHighlighter>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {`// the operating branch:\n-b [ --branch ] arg`}
        </SyntaxHighlighter>
        <Typography component="p">
          Utility Options:
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'// none'}
        </SyntaxHighlighter>
        <br />

        <Typography variant="h5" gutterBottom>
          Examples
        </Typography>
        <SyntaxHighlighter language='javascript' style={gruvboxDark}>
          {`
ustore> create_dataset -t sampleDS1 -b master
[SUCCESS: CREATE_DATASET] Dataset "sampleDS1" has been created for Branch "master"

ustore> create_dataset -t DS2 -b master
[SUCCESS: CREATE_DATASET] Dataset "DS2" has been created for Branch "master"

ustore> create_dataset -t sampleDS1 -b newFeature
[SUCCESS: CREATE_DATASET] Dataset "sampleDS1" has been created for Branch "newFeature"

ustore> create_dataset -t DS2 -b master
[FAILED: CREATE_DATASET] Dataset: "DS2", Branch: "master" --> Error(13): branch already exists
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
