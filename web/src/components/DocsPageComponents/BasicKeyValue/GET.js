import React from 'react';
import { withStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardContent from '@material-ui/core/CardContent';
import Typography from '@material-ui/core/Typography';

import SyntaxHighlighter from 'react-syntax-highlighter';
import { gruvboxDark, solarizedLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';

import { GeneralOptions, UtilityOptions } from "../Options"

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
          Get
        </Typography>
        <Typography component="p">
          Get the value based on the key
        </Typography>
        <br />

        <Typography variant="h5" gutterBottom>
          Syntax
        </Typography>
        <SyntaxHighlighter language='javascript' style={solarizedLight}>
          {'GET{_ALL} -k <key> [-b <branch> | -v <version>] {<file>}'}
        </SyntaxHighlighter>
        <Typography component="p">
          Parameters:
        </Typography>
        {GeneralOptions._k}
        {GeneralOptions._b}
        {GeneralOptions._v}
        {GeneralOptions._file}
        <Typography component="p">
          Utility Options:
        </Typography>
        {UtilityOptions._1}
        <br />

        <Typography variant="h5" gutterBottom>
          Examples
        </Typography>
        <SyntaxHighlighter language='javascript' style={gruvboxDark}>
          {`
ustore> get -k myfirstKey -b master
[SUCCESS: GET] Value<Blob>: "this is the first key i put into the rafiki"

ustore> get -k myfirstKey -v A62JUGB6ORFDMN3LKMFDM6MRLAKMFIXE
[SUCCESS: GET] Value<Blob>: "i want to know the version generated"

// providing the <file> parameter
// will save the value as specified in the <file>
ustore> get -k File1 -b master ./download.dat
[SUCCESS: GET] Value<Blob>: --> ./download.dat  [129B]

// Get_ALL is to display full value from map, list, set
// e.g. get_all -k someMap -b master
          `}
        </SyntaxHighlighter>
      </CardContent>
    </Card>
  );
}


export default withStyles(styles)(DocsCard);
